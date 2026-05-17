from .bark import BarkScale
from .erb import ErbScale
from .octave import OctaveScale

# Ready-to-use instances with a unified method surface.
bark = BarkScale()
erb = ErbScale()
octave = OctaveScale()

# Backwards-compatible instance aliases.
bark_scale = bark
erb_scale = erb
octave_scale = octave

__all__ = [
	"BarkScale",
	"ErbScale",
	"OctaveScale",
	"bark",
	"erb",
	"octave",
	"bark_scale",
	"erb_scale",
	"octave_scale",
]
