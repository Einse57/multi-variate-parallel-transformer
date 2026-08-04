"""iEEG data source abstraction — h5py-compatible wrappers for EDF/EDF+.

``open_patient_file(path)`` returns an object that supports the same
dict-like access pattern used by ``LongTermEEGDataset``:

    f["data/ieeg"]          →  dataset-like with .shape and .read_direct()
    f["data/seizures"][:]   →  structured array with 'onsets'/'offsets'
    f.attrs["sampling_rate"]→  float

For H5 files this is ``h5py.File``, or ``_H5SuperContactsFile`` when
``super_contacts=True`` is passed.
For EDF/EDF+ files this is an ``EDFPatientFile`` adapter.

Preprocessing for EDF:
    The SWEC-ETHZ H5 dataset is already preprocessed (bandpass 0.5–120 Hz,
    median re-referenced).  Raw EDF files typically are NOT.  Pass
    ``preprocess=True`` (default) to ``EDFPatientFile`` or
    ``open_patient_file()`` to apply the same pipeline automatically.

Super-contacts:
    Pass ``super_contacts=True`` (and optionally ``super_contacts_target``)
    to ``open_patient_file()`` to average adjacent channel groups, reducing
    the channel count while preserving LFP signal.  Works for both H5 and
    EDF sources.  See ``pair_average_channels`` for details.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


# ── Local averaging (Super-Contacts) ──────────────────────────────────────
#
# Spatial rationale: adjacent contact pairs share correlated LFP signal while
# their thermal/amplifier noise is independent.  Averaging N contacts improves
# SNR by √N (Buzsáki, Anastassiou & Koch, Nat Rev Neurosci 13:407, 2012).
# For macro-contacts spaced < spatial Nyquist (~1.25 mm), decimation alone
# discards real focal generators; averaging preserves them in the sum while
# cancelling uncorrelated noise (Slutzky et al., J Neural Eng 7:026004, 2010).
#
# Signal character: the averaged output stays in the same physical units and
# frequency band as median-referenced LFP — no remontaging artefact, no
# high-pass bias — making it distribution-compatible with SWEZ-ETHZ training
# data (Burrello et al., 2019; original SWEZ preprocessing: bandpass 0.5-120 Hz,
# median re-reference).


def pair_average_channels(data: np.ndarray, target: Optional[int] = None) -> np.ndarray:
    """Average adjacent channel groups to reduce channel count.

    Uses np.array_split so channels are distributed into target bins as evenly
    as possible (no padding, no duplicated boundary channels).  For C channels
    split into T bins: (C % T) bins get ceil(C/T) channels, the rest get
    floor(C/T).  SNR improves by √group_size (Buzsáki et al., 2012).

    Args:
        data:   (channels, samples) float array.
        target: desired output channel count.  Defaults to ceil(C/2).

    Returns:
        (target, samples) array.
        If the source already has ≤ target channels, returns the source unchanged.
    """
    n_ch = data.shape[0]

    if target is None:
        target = math.ceil(n_ch / 2)

    if target <= 0:
        raise ValueError(f"target must be a positive integer, got {target}")

    if n_ch <= target:
        return data

    # Accumulate in float32 to avoid precision loss from averaging float16 inputs.
    work = data.astype(np.float32, copy=False)
    groups = np.array_split(work, target, axis=0)
    return np.stack([g.mean(axis=0) for g in groups]).astype(data.dtype)


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


# ── Super-contacts lazy wrapper ───────────────────────────────────────────

class _SuperContactsDataset:
    """Wraps any dataset-like (h5py.Dataset or _ArrayDataset) and applies
    pair_average_channels on every read, reducing the channel dimension.

    Downstream code sees a dataset with shape (target, T) without needing
    to know whether the source is H5 or EDF.
    """

    def __init__(self, inner, target: Optional[int] = None):
        self._inner = inner
        raw_ch = inner.shape[0]
        self._out_ch = target if target is not None else math.ceil(raw_ch / 2)
        # Pass-through when the source is already at or below the requested count.
        self._passthrough = raw_ch <= self._out_ch
        self._shape = (raw_ch if self._passthrough else self._out_ch,) + inner.shape[1:]

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._inner.dtype

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        # Read all channels for the requested time slice, then average.
        if self._passthrough:
            return self._inner.read_direct(dest, source_sel, dest_sel)
        if source_sel is None:
            raw = self._inner[:]
        else:
            # source_sel is typically np.s_[:, t_start:t_end]
            raw = self._inner[source_sel]
        averaged = pair_average_channels(raw, target=self._out_ch)
        if dest_sel is None:
            dest[:] = averaged
        else:
            dest[dest_sel] = averaged

    def __getitem__(self, key):
        raw = self._inner[key]
        if self._passthrough or raw.ndim != 2:
            return raw
        return pair_average_channels(raw, target=self._out_ch)


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

    Preprocessing (``preprocess`` parameter):
        SWEC-ETHZ H5 files are already preprocessed (bandpass filtered
        0.5–120 Hz, median re-referenced).  Raw EDF recordings are not.

        ``preprocess="auto"`` (default):
            Detects whether each step has already been applied by
            checking DC offset ratio and cross-channel median.  Steps
            that appear already done are skipped automatically, so
            pre-filtered EDF files are not double-processed.
        ``preprocess=True``:
            Forces all steps unconditionally.
        ``preprocess=False``:
            Skips preprocessing entirely (use when handling it externally).

        Steps (when applied):
          1. Bandpass 0.5–120 Hz (4th-order Butterworth, zero-phase)
          2. Optional notch filter (50 or 60 Hz) for powerline noise
          3. Median re-reference (subtract cross-channel median)
          4. Resample to 512 Hz
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
        preprocess: bool | str = "auto",
        notch_freq: Optional[float] = None,
        target_sfreq: float = 512.0,
        super_contacts: bool = False,
        super_contacts_target: Optional[int] = None,
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

        # ── Apply SWEC-paper preprocessing if requested ────────────────
        #   preprocess="auto" : detect whether steps are needed (default)
        #   preprocess=True   : force all steps
        #   preprocess=False  : skip entirely
        if preprocess is not False:
            from .edf_preprocess import preprocess_edf_for_mvpformer

            auto = (preprocess == "auto")
            data, sfreq = preprocess_edf_for_mvpformer(
                data,
                sfreq,
                target_sfreq=target_sfreq,
                bandpass=True,
                bandpass_low=0.5,
                bandpass_high=120.0,
                bandpass_order=4,
                notch=notch_freq,
                do_median_ref=True,
                do_resample=True,
                auto_detect=auto,
            )
            print(
                f"  ✓ EDF preprocessing applied (mode={'auto' if auto else 'force'}): "
                f"resampled to {sfreq} Hz"
                + (f", notch {notch_freq} Hz" if notch_freq else "")
            )

        # Build seizure structured array from EDF+ annotations
        seizures = self._parse_seizure_annotations(raw.annotations)

        if super_contacts:
            data = pair_average_channels(data, target=super_contacts_target)
            print(f"  ✓ Super-contacts applied: {data.shape[0]} averaged channels")

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


class _H5SuperContactsFile:
    """Thin proxy around h5py.File that returns a _SuperContactsDataset for
    'data/ieeg' while passing all other key lookups straight through.

    h5py.File is read-only and does not support item assignment, so we wrap
    it rather than patch it.
    """

    def __init__(self, h5file, target: Optional[int] = None):
        self._f = h5file
        self._target = target
        self._wrapped = _SuperContactsDataset(h5file["data/ieeg"], target=target)

    def __getitem__(self, key: str):
        if key == "data/ieeg":
            return self._wrapped
        return self._f[key]

    def __contains__(self, key: str) -> bool:
        return key in self._f

    @property
    def attrs(self):
        return self._f.attrs

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_patient_file(path: str | Path, **kwargs):
    """Open a patient recording file, returning an h5py.File-compatible object.

    For H5 files:  returns ``h5py.File``, or ``_H5SuperContactsFile`` when
                   ``super_contacts=True`` is in kwargs.
    For EDF files: returns ``EDFPatientFile`` adapter.

    Extra kwargs are forwarded to the constructor (e.g. ``picks`` for EDF,
    ``preprocess=True/False``, ``notch_freq``, ``super_contacts``,
    ``super_contacts_target``).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in _H5_EXTENSIONS:
        try:
            import hdf5plugin  # noqa: F401
        except ImportError:
            pass
        import h5py
        sc = kwargs.pop("super_contacts", False)
        sc_target = kwargs.pop("super_contacts_target", None)
        # Strip EDF-only kwargs before passing to h5py
        h5_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("picks", "preprocess", "notch_freq", "target_sfreq")}
        f = h5py.File(str(p), "r", **h5_kwargs)
        if sc:
            return _H5SuperContactsFile(f, target=sc_target)
        return f

    if ext in _EDF_EXTENSIONS:
        return EDFPatientFile(p, **kwargs)

    supported = ", ".join(sorted(_SUPPORTED))
    raise ValueError(
        f"Unsupported file format '{ext}' for {p.name}. Supported: {supported}"
    )
