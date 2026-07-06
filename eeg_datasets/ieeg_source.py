"""
iEEG data source abstraction — h5py-compatible wrappers for EDF/EDF+.

``open_patient_file(path)`` returns an object that supports the same
dict-like access pattern used by ``LongTermEEGDataset``:

    f["data/ieeg"]          →  dataset-like with .shape and .read_direct()
    f["data/seizures"][:]   →  structured array with 'onsets'/'offsets'
    f.attrs["sampling_rate"]→  float

For H5 files this is just ``h5py.File``.
For EDF/EDF+ files this is an ``EDFPatientFile`` adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ── Dataset-like wrapper for in-memory arrays ─────────────────────────────

class _ArrayDataset:
    """Mimics the subset of h5py.Dataset used by LongTermEEGDataset.

    Supports ``.shape`` and ``.read_direct(dest, source_sel, dest_sel)``
    so existing ``__getitem__`` code works unchanged.
    """

    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        if source_sel is None:
            source_sel = ()
        if dest_sel is None:
            dest_sel = ()
        dest[dest_sel] = self._data[source_sel]

    def __getitem__(self, key):
        return self._data[key]


# ── Attrs-like wrapper ────────────────────────────────────────────────────

class _Attrs(dict):
    """Dict that also works as h5py attrs (supports __getitem__)."""
    pass


# ── EDF/EDF+ adapter ─────────────────────────────────────────────────────

class EDFPatientFile:
    """Wraps an EDF/EDF+ file to present the same interface as an
    h5py.File opened on a SWEZ-ETHZ H5 patient recording.

    On construction the entire file is read into memory via MNE
    (EDF files are typically much shorter than multi-day H5 recordings,
    so this is acceptable).

    Channel selection:
        By default only EEG-typed channels are kept.  Pass ``picks=None``
        to keep everything, or a list of channel names.

    Non-EEG channel guard:
        Channels whose names match common non-EEG patterns (ECG, EMG,
        EOG, SpO2, etc.) are dropped automatically, even if the EDF
        header marks them as EEG.  This catches mislabelled files.
    """

    # Patterns that indicate a channel is NOT brain signal.
    # Matched case-insensitively against channel name.
    _NON_EEG_PATTERNS = (
        "ecg", "ekg", "emg", "eog",
        "spo2", "sp02",            # pulse oximetry (note: O vs 0)
        "resp", "airflow", "thorax", "abdomen",
        "snore", "pleth",
        "hr", "pulse", "temp",
        "trigger", "event", "stim", "mark",
        "dc", "ref",               # DC channels, standalone ref
    )

    def __init__(
        self,
        path: str | Path,
        *,
        picks: Optional[str | list] = "eeg",
    ):
        import mne

        self._path = Path(path)
        raw = mne.io.read_raw_edf(str(self._path), preload=True, verbose=False)

        # Channel type selection (MNE-based)
        if picks is not None:
            try:
                raw.pick(picks)
            except ValueError:
                pass  # keep all if pick type fails

        # Name-based guard: drop channels matching non-EEG patterns
        drop = [
            ch for ch in raw.ch_names
            if self._is_non_eeg_name(ch)
        ]
        if drop:
            raw.drop_channels(drop)
            print(f"  ⚠ Dropped {len(drop)} non-EEG channel(s): {drop}")

        if raw.info["nchan"] == 0:
            raise ValueError(
                f"No channels remaining after filtering in {self._path}. "
                f"Try picks=None to keep all channels."
            )

        sfreq = float(raw.info["sfreq"])
        data = raw.get_data().astype(np.float32)  # (channels, samples)

        # Build seizure structured array from EDF+ annotations
        seizures = self._parse_seizure_annotations(raw.annotations)

        # Store as dict-like datasets
        self._datasets = {
            "data/ieeg": _ArrayDataset(data),
            "data/seizures": seizures,
        }
        self.attrs = _Attrs({"sampling_rate": sfreq})

    def __getitem__(self, key: str):
        if key in self._datasets:
            return self._datasets[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._datasets

    def close(self):
        pass  # in-memory, nothing to close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def _is_non_eeg_name(cls, ch_name: str) -> bool:
        """Return True if *ch_name* matches a known non-EEG pattern."""
        lower = ch_name.strip().lower()
        for pat in cls._NON_EEG_PATTERNS:
            # Exact match, or pattern followed by separator/digit
            if lower == pat:
                return True
            if lower.startswith(pat) and (
                len(lower) == len(pat)
                or lower[len(pat)] in " -_0123456789"
            ):
                return True
        return False

    @staticmethod
    def _parse_seizure_annotations(annotations) -> np.ndarray:
        """Parse MNE annotations into a structured array matching SWEZ format.

        Returns structured array with 'onsets' and 'offsets' fields (seconds),
        or an empty array if no seizure annotations are found.
        """
        _SZ_KEYWORDS = {"seizure", "sz", "ictal"}

        onsets = []
        offsets = []
        for annot in annotations:
            desc = annot["description"].lower()
            if any(kw in desc for kw in _SZ_KEYWORDS):
                onset = float(annot["onset"])
                duration = float(annot["duration"])
                if duration > 0:
                    onsets.append(onset)
                    offsets.append(onset + duration)

        if not onsets:
            # Return empty structured array with correct dtype
            dtype = np.dtype([("onsets", "<f8"), ("offsets", "<f8")])
            return np.array([], dtype=dtype).reshape(0)

        dtype = np.dtype([("onsets", "<f8"), ("offsets", "<f8")])
        arr = np.empty(len(onsets), dtype=dtype)
        arr["onsets"] = onsets
        arr["offsets"] = offsets
        return arr


# ── Factory ───────────────────────────────────────────────────────────────

_EDF_EXTENSIONS = {".edf"}
_H5_EXTENSIONS = {".h5", ".hdf5"}
_SUPPORTED = _EDF_EXTENSIONS | _H5_EXTENSIONS


def open_patient_file(path: str | Path, **kwargs):
    """Open a patient recording file, returning an h5py.File-compatible object.

    For H5 files:  returns ``h5py.File`` directly.
    For EDF files: returns ``EDFPatientFile`` adapter.

    Extra kwargs are forwarded to the constructor (e.g. ``picks`` for EDF).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _H5_EXTENSIONS:
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import h5py
        return h5py.File(str(p), "r", **kwargs)

    if ext in _EDF_EXTENSIONS:
        return EDFPatientFile(p, **kwargs)

    supported = ", ".join(sorted(_SUPPORTED))
    raise ValueError(
        f"Unsupported file format '{ext}' for {p.name}. Supported: {supported}"
    )
