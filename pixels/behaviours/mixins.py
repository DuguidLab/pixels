"""
Mixin classes to extend functionality in user scripts.
"""
from pixels import PixelsError


class ProbeDepthMixin:
    def set_probe_depth(self, depth: float) -> None:
        self._probe_depth = depth

    def get_probe_depth(self) -> float:
        if not hasattr(self, "_probe_depth"):
            raise PixelsError("Call set_probe_depth first")
        return self._probe_depth
