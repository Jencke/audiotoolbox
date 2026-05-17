from .bark import BarkScale
from .erb import ErbScale
from .greenwood import GreenwoodScale
from .mel import MelScale
from .octave import OctaveScale
from .semitone import SemitoneScale

# Ready-to-use instances with a unified method surface.
bark = BarkScale()
erb = ErbScale()
greenwood = GreenwoodScale()
mel = MelScale()
octave = OctaveScale()
semitone = SemitoneScale()

# Backwards-compatible instance aliases.
bark_scale = bark
erb_scale = erb
greenwood_scale = greenwood
mel_scale = mel
octave_scale = octave
semitone_scale = semitone

__all__ = [
	"BarkScale",
	"ErbScale",
	"GreenwoodScale",
	"MelScale",
	"OctaveScale",
	"SemitoneScale",
	"bark",
	"erb",
	"greenwood",
	"mel",
	"octave",
	"semitone",
	"bark_scale",
	"erb_scale",
	"greenwood_scale",
	"mel_scale",
	"octave_scale",
	"semitone_scale",
]
