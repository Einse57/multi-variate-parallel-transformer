"""XPU Accelerator for PyTorch Lightning 2.6.x.

Registers an 'xpu' accelerator so that PL Trainer can use Intel XPU devices
(Meteor Lake iGPU, Arc, etc.) via ``accelerator='xpu'`` or ``accelerator='auto'``.

Usage — import this module before constructing the Trainer / CLI:
    import xpu_accelerator  # noqa: F401
"""

import torch
from typing import Any, Optional, Union
from typing_extensions import override

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators import AcceleratorRegistry
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.types import _DEVICE


class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise RuntimeError(f"Device should be XPU, got {device} instead")
        torch.xpu.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        torch.xpu.empty_cache()

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        return {}

    @override
    def teardown(self) -> None:
        torch.xpu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        if devices is None or devices == "auto":
            return list(range(torch.xpu.device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            # "0,1" → specific device IDs; "2" → count (= use 2 devices)
            if "," in devices:
                return [int(x.strip()) for x in devices.split(",")]
            return list(range(int(devices)))
        # list of ints → specific device IDs
        return list(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "xpu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )


# ---------------------------------------------------------------------------
# Self-register on import
# ---------------------------------------------------------------------------
if "xpu" not in AcceleratorRegistry:
    XPUAccelerator.register_accelerators(AcceleratorRegistry)

# Also register in the fabric-level registry so _select_auto_accelerator can
# be patched to find us.
try:
    from lightning_fabric.accelerators import _ACCELERATORS_BASE_MODULE  # noqa: F401
except ImportError:
    pass

# Monkey-patch the auto-selection so accelerator='auto' picks XPU when available.
import lightning_fabric.utilities.device_parser as _dp

_orig_select = _dp._select_auto_accelerator


def _select_auto_accelerator_with_xpu() -> str:
    if XPUAccelerator.is_available():
        return "xpu"
    return _orig_select()


_dp._select_auto_accelerator = _select_auto_accelerator_with_xpu

# Also patch the connector module's own reference (it was imported by name
# before our patch, so the module-level binding is stale).
import pytorch_lightning.trainer.connectors.accelerator_connector as _ac_mod

_ac_mod._select_auto_accelerator = _select_auto_accelerator_with_xpu

# Also need to patch the AcceleratorConnector so it knows how to handle
# accelerator='xpu' string → XPUAccelerator class.
from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector

_orig_check = _AcceleratorConnector._check_config_and_set_final_flags


def _patched_check(self, *args, **kwargs):
    _orig_check(self, *args, **kwargs)


# Patch _AcceleratorConnector to recognize 'xpu' as a valid accelerator.
_orig_init = _AcceleratorConnector.__init__


def _patched_ac_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    # If the accelerator ended up as 'cpu' but XPU is available and was
    # requested (or auto), swap it in.
    if isinstance(self._accelerator_flag, str) and self._accelerator_flag == "xpu":
        pass  # The registry lookup will handle it


_AcceleratorConnector.__init__ = _patched_ac_init

# Patch _choose_strategy so single-device XPU gets device='xpu:0' instead of 'cpu'.
_orig_choose_strategy = _AcceleratorConnector._choose_strategy


def _patched_choose_strategy(self):
    strategy = _orig_choose_strategy(self)
    # If we got a SingleDeviceStrategy stuck on 'cpu' but the accelerator is XPU,
    # fix the root device.
    from pytorch_lightning.strategies import SingleDeviceStrategy

    if (
        isinstance(strategy, SingleDeviceStrategy)
        and strategy.root_device.type == "cpu"
        and isinstance(self.accelerator, XPUAccelerator)
        and self._parallel_devices
    ):
        device = self._parallel_devices[0]
        strategy._root_device = device
    return strategy


_AcceleratorConnector._choose_strategy = _patched_choose_strategy
