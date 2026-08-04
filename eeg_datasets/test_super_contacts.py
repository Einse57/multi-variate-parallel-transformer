"""Tests for pair_average_channels and _SuperContactsDataset."""
import math

import numpy as np
import pytest

from eeg_datasets.ieeg_source import (
    _ArrayDataset,
    _SuperContactsDataset,
    pair_average_channels,
)

SAMPLES = 100


def _data(n_ch: int, dtype=np.float32) -> np.ndarray:
    """Deterministic channel data: channel i has constant value i."""
    return np.tile(np.arange(n_ch, dtype=dtype)[:, None], (1, SAMPLES))


# ── pair_average_channels ─────────────────────────────────────────────────


class TestPairAverageChannels:
    def test_256_to_128_pairs(self):
        data = _data(256)
        out = pair_average_channels(data, target=128)
        assert out.shape == (128, SAMPLES)
        # Each output i should be mean of channels 2i and 2i+1.
        for i in range(128):
            assert out[i, 0] == pytest.approx((2 * i + 2 * i + 1) / 2)

    def test_512_to_128_groups_of_4(self):
        data = _data(512)
        out = pair_average_channels(data, target=128)
        assert out.shape == (128, SAMPLES)
        # Group 0: channels 0,1,2,3 → mean = 1.5
        assert out[0, 0] == pytest.approx(1.5)
        # All 512 channels contribute — last group mean = mean(508,509,510,511)
        assert out[127, 0] == pytest.approx((508 + 509 + 510 + 511) / 4)

    def test_401_to_128_no_padding_duplicates(self):
        data = _data(401)
        out = pair_average_channels(data, target=128)
        assert out.shape == (128, SAMPLES)
        # array_split gives 17 groups of 4 and 111 groups of 3.
        # Last output channel must not be a duplicate of channel 400.
        last_group_mean = np.mean(np.arange(401 - 3, 401))  # floor split
        assert out[127, 0] == pytest.approx(last_group_mean)

    def test_128_passthrough(self):
        data = _data(128)
        out = pair_average_channels(data, target=128)
        assert out.shape == (128, SAMPLES)
        np.testing.assert_array_equal(out, data)

    def test_127_passthrough(self):
        data = _data(127)
        out = pair_average_channels(data, target=128)
        assert out.shape == (127, SAMPLES)
        np.testing.assert_array_equal(out, data)

    def test_64_passthrough(self):
        data = _data(64)
        out = pair_average_channels(data, target=128)
        np.testing.assert_array_equal(out, data)

    def test_1_channel_passthrough(self):
        data = _data(1)
        out = pair_average_channels(data, target=128)
        np.testing.assert_array_equal(out, data)

    def test_129_to_128_asymmetric(self):
        data = _data(129)
        out = pair_average_channels(data, target=128)
        assert out.shape == (128, SAMPLES)
        # array_split(129, 128): first group has 2 channels, rest have 1.
        assert out[0, 0] == pytest.approx(0.5)   # mean(0, 1)
        assert out[1, 0] == pytest.approx(2.0)    # channel 2 alone
        assert out[127, 0] == pytest.approx(128.0)  # channel 128 alone

    def test_no_target_default_halves(self):
        data = _data(256)
        out = pair_average_channels(data)  # no target → ceil(256/2) = 128
        assert out.shape == (128, SAMPLES)

    def test_no_target_odd_channels_gives_ceil(self):
        # 257 channels, no target → ceil(257/2) = 129, NOT 128.
        # This is the footgun: users must set super_contacts_target explicitly.
        data = _data(257)
        out = pair_average_channels(data)
        assert out.shape == (129, SAMPLES)

    def test_float16_input_no_precision_collapse(self):
        # Averaging four float16 values that are close but not identical should
        # not collapse to a single value due to float16 accumulation error.
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (512, SAMPLES)).astype(np.float16)
        out = pair_average_channels(data, target=128)
        assert out.dtype == np.float16
        # Variance of output should be non-zero (not all collapsed to same value).
        assert out[:, 0].std() > 0

    def test_float16_result_dtype_preserved(self):
        data = _data(256, dtype=np.float16)
        out = pair_average_channels(data, target=128)
        assert out.dtype == np.float16

    def test_target_zero_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            pair_average_channels(_data(256), target=0)

    def test_target_negative_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            pair_average_channels(_data(256), target=-1)

    def test_output_values_use_all_channels(self):
        # Verify no channels are silently dropped: sum of all output means ×
        # their group sizes must equal sum of all input channel values.
        data = _data(401)
        out = pair_average_channels(data, target=128)
        groups = np.array_split(np.arange(401, dtype=np.float32), 128)
        expected = np.array([g.mean() for g in groups])
        np.testing.assert_allclose(out[:, 0], expected, rtol=1e-5)


# ── _SuperContactsDataset ─────────────────────────────────────────────────


class TestSuperContactsDataset:
    def _wrap(self, n_ch: int, target: int) -> _SuperContactsDataset:
        return _SuperContactsDataset(_ArrayDataset(_data(n_ch)), target=target)

    def test_shape_reported_correctly_256(self):
        ds = self._wrap(256, 128)
        assert ds.shape == (128, SAMPLES)

    def test_shape_reported_correctly_passthrough(self):
        ds = self._wrap(128, 128)
        assert ds.shape == (128, SAMPLES)

    def test_getitem_averaging(self):
        ds = self._wrap(256, 128)
        out = ds[:]
        assert out.shape == (128, SAMPLES)

    def test_getitem_passthrough(self):
        data = _data(128)
        ds = _SuperContactsDataset(_ArrayDataset(data), target=128)
        np.testing.assert_array_equal(ds[:], data)

    def test_read_direct_averaging(self):
        ds = self._wrap(256, 128)
        dest = np.zeros((128, SAMPLES), dtype=np.float32)
        ds.read_direct(dest, source_sel=np.s_[:, :], dest_sel=np.s_[:, :])
        assert dest.shape == (128, SAMPLES)
        assert dest[0, 0] == pytest.approx(0.5)  # mean(0, 1)

    def test_read_direct_passthrough_delegates(self):
        data = _data(64)
        ds = _SuperContactsDataset(_ArrayDataset(data), target=128)
        dest = np.zeros((64, SAMPLES), dtype=np.float32)
        ds.read_direct(dest)
        np.testing.assert_array_equal(dest, data)

    def test_passthrough_flag_set_correctly(self):
        assert self._wrap(256, 128)._passthrough is False
        assert self._wrap(128, 128)._passthrough is True
        assert self._wrap(64, 128)._passthrough is True
