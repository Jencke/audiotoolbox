from __future__ import annotations

from abc import ABC, abstractmethod


class ScaleBase(ABC):
    """Common interface for auditory scales.

    Implementations should convert between frequency and scale units and
    provide bandwidth values in Hz from a center frequency.
    """

    @abstractmethod
    def from_freq(self, frequency, **kwargs):
        """Convert frequency in Hz to scale value."""

    @abstractmethod
    def to_freq(self, scale_value, **kwargs):
        """Convert scale value to frequency in Hz."""

    @abstractmethod
    def get_bw(self, fc, **kwargs):
        """Calculate bandwidth in Hz for a center frequency."""

    def calc_bw(self, fc, **kwargs):
        """Compatibility alias for get_bw."""
        return self.get_bw(fc, **kwargs)
