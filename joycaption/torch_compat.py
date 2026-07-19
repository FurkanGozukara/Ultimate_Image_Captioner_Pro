from __future__ import annotations

from enum import Enum
from typing import Any


def install_torchao_enum_compatibility() -> bool:
    """Avoid TorchAO's obsolete Enum pytree registration on newer PyTorch.

    PyTorch now treats Enum subclasses as opaque values natively. TorchAO 0.18
    still decorates two Enums with ``register_constant``; PyTorch 2.13 warns and
    its announced next step is an exception. This compatibility bridge changes
    that deprecated call into the native no-op while preserving registration for
    every non-Enum class.
    """

    try:
        import torch
        from torch._library.opaque_object import is_opaque_type
    except (ImportError, AttributeError):
        return False

    pytree = torch.utils._pytree
    current = pytree.register_constant
    if getattr(current, "_joycaption_enum_compat", False):
        return True

    def register_constant_compat(cls: type[Any], *args: Any, **kwargs: Any) -> None:
        if isinstance(cls, type) and issubclass(cls, Enum) and is_opaque_type(cls):
            return None
        return current(cls, *args, **kwargs)

    register_constant_compat._joycaption_enum_compat = True  # type: ignore[attr-defined]
    register_constant_compat._joycaption_original = current  # type: ignore[attr-defined]
    pytree.register_constant = register_constant_compat
    return True
