from __future__ import annotations

from typing import Optional

import numpy as np

from . import core as audio
from .signal import Signal


def _spherical_to_cartesian(azimuth, elevation):
    r"""Convert spherical source directions to cartesian unit vectors.

    Uses the SOFA convention: ``azimuth`` increases counter-clockwise in the
    horizontal plane, ``elevation`` is measured from the horizontal plane.
    Both are given in degrees. The returned vectors lie on the unit sphere,
    so any measurement distance (radius) is intentionally ignored.

    Parameters
    ----------
    azimuth : array_like
        Azimuth angle(s) in degrees.
    elevation : array_like
        Elevation angle(s) in degrees.

    Returns
    -------
    ndarray
        Unit vectors of shape ``(n, 3)``.
    """
    az = np.deg2rad(np.atleast_1d(azimuth))
    el = np.deg2rad(np.atleast_1d(elevation))
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.stack([x, y, z], axis=-1)


class HRIRSet:
    r"""A set of head-related impulse responses measured over directions.

    The impulse responses are stored time-domain in a
    :class:`audiotoolbox.Signal` of shape ``(n_taps, n_directions, 2)`` where
    the last axis holds the (left, right) ear. The associated source
    directions are kept in a separate position table.

    Parameters
    ----------
    hrirs : Signal
        Time-domain impulse responses of shape ``(n_taps, n_directions, 2)``.
    positions : array_like
        Source directions of shape ``(n_directions, 2)`` or
        ``(n_directions, 3)`` given as ``(azimuth, elevation[, distance])``.
        Angles are in degrees.
    coordinate_system : str, optional
        Name of the coordinate system the positions are given in
        (default = ``"spherical"``).
    units : str, optional
        Free-form description of the position units (e.g. the SOFA
        ``SourcePosition_Units`` string). Stored for reference only.

    Examples
    --------
    >>> import numpy as np, audiotoolbox as audio
    >>> hrirs = audio.Signal((4, 2), 128 / 48000, 48000)  # 4 directions, L/R
    >>> positions = np.array([[0, 0], [90, 0], [180, 0], [270, 0]])
    >>> hrir_set = audio.HRIRSet(hrirs, positions)
    >>> hrir_set.n_directions
    4
    """

    def __init__(
        self,
        hrirs: Signal,
        positions,
        coordinate_system: str = "spherical",
        units: Optional[str] = None,
    ):
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError(
                "positions must have shape (n_directions, 2) or "
                "(n_directions, 3), got "
                f"{positions.shape}."
            )

        channel_shape = hrirs.channel_shape
        if len(channel_shape) != 2 or channel_shape[-1] != 2:
            raise ValueError(
                "hrirs must be a Signal of shape (n_taps, n_directions, 2); "
                f"got channel shape {channel_shape}."
            )
        if channel_shape[0] != len(positions):
            raise ValueError(
                f"Number of directions does not match: {channel_shape[0]} "
                f"HRIRs but {len(positions)} positions."
            )

        self.hrirs = hrirs
        self.positions = positions
        self.coordinate_system = coordinate_system
        self.units = units

        # Cached geometry, built on demand (see _unit_vectors / _hull).
        self._unit_vectors_cache = None
        self._hull_cache = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_sofa(cls, filename: str) -> "HRIRSet":
        """Load an HRIR set from a SOFA file.

        Reads a SOFA file (the standard interchange format for HRTFs) using
        the :mod:`sofar` library. The convention is expected to store the
        impulse responses in ``Data_IR`` with shape
        ``(n_directions, 2, n_taps)`` (e.g. ``SimpleFreeFieldHRIR``).

        Parameters
        ----------
        filename : str
            Path to the ``.sofa`` file.

        Returns
        -------
        HRIRSet
            The loaded HRIR set.

        Raises
        ------
        ImportError
            If the optional :mod:`sofar` dependency is not installed.
        ValueError
            If the file does not provide two-receiver impulse responses.
        """
        try:
            import sofar
        except ImportError as exc:  # pragma: no cover - exercised via message
            raise ImportError(
                "Reading SOFA files requires the optional 'sofar' package. "
                "Install it with `pip install sofar` or "
                "`pip install audiotoolbox[hrtf]`."
            ) from exc

        sofa = sofar.read_sofa(filename)

        # Data_IR: (n_directions, n_receivers, n_taps)
        ir = np.asarray(sofa.Data_IR, dtype=float)
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError(
                "Expected Data_IR with shape (n_directions, 2, n_taps), got "
                f"{ir.shape}. Only two-ear HRIR conventions are supported."
            )

        fs = int(np.asarray(sofa.Data_SamplingRate).flatten()[0])
        n_directions, _, n_taps = ir.shape

        hrirs = Signal((n_directions, 2), n_taps / fs, fs)
        # reorder receivers/taps -> (n_taps, n_directions, 2)
        hrirs[:] = np.moveaxis(ir, -1, 0)

        positions = np.asarray(sofa.SourcePosition, dtype=float)
        coordinate_system = str(getattr(sofa, "SourcePosition_Type", "spherical"))
        units = getattr(sofa, "SourcePosition_Units", None)
        if units is not None:
            units = str(units)

        return cls(hrirs, positions, coordinate_system, units)

    # ------------------------------------------------------------------
    # cheap derived views
    # ------------------------------------------------------------------
    @property
    def fs(self) -> int:
        """Sampling rate in Hz."""
        return self.hrirs.fs

    @property
    def n_directions(self) -> int:
        """Number of measured directions."""
        return self.hrirs.channel_shape[0]

    @property
    def n_taps(self) -> int:
        """Length of each impulse response in samples."""
        return self.hrirs.n_samples

    @property
    def azimuth(self) -> np.ndarray:
        """Azimuth of every measured direction in degrees."""
        return self.positions[:, 0]

    @property
    def elevation(self) -> np.ndarray:
        """Elevation of every measured direction in degrees."""
        return self.positions[:, 1]

    @property
    def distance(self) -> Optional[np.ndarray]:
        """Measurement distance per direction, or ``None`` if not stored."""
        if self.positions.shape[1] < 3:
            return None
        return self.positions[:, 2]

    @property
    def _unit_vectors(self) -> np.ndarray:
        """Cartesian unit vectors of all measured directions."""
        if self._unit_vectors_cache is None:
            self._unit_vectors_cache = _spherical_to_cartesian(
                self.azimuth, self.elevation
            )
        return self._unit_vectors_cache

    def to_hrtf(self):
        """Return the frequency-domain transfer functions.

        Converts the stored impulse responses to the frequency domain. As
        with :meth:`audiotoolbox.Signal.to_freqdomain` this is not done in
        place; a new :class:`audiotoolbox.FrequencyDomainSignal` of shape
        ``(n_taps, n_directions, 2)`` is returned.

        Returns
        -------
        FrequencyDomainSignal
            The head-related transfer functions.
        """
        return self.hrirs.to_freqdomain()

    # ------------------------------------------------------------------
    # spatial lookup
    # ------------------------------------------------------------------
    def _nearest_index(self, azimuth, elevation) -> int:
        """Index of the measured direction closest to the query direction."""
        query = _spherical_to_cartesian(azimuth, elevation)[0]
        # Nearest on the unit sphere == largest dot product (cosine distance).
        dots = self._unit_vectors @ query
        return int(np.argmax(dots))

    def nearest(self, azimuth: float, elevation: float) -> Signal:
        """Return the HRIR of the closest measured direction.

        Parameters
        ----------
        azimuth : float
            Query azimuth in degrees.
        elevation : float
            Query elevation in degrees.

        Returns
        -------
        Signal
            A two-channel (left, right) signal of shape ``(n_taps, 2)``.
        """
        idx = self._nearest_index(azimuth, elevation)
        return self.hrirs.ch[idx].copy()

    def _barycentric_weights_3d(self, query):
        """Enclosing-triangle indices/weights for a fully 3-D measurement grid.

        Triangulates the measured directions on the unit sphere (via their
        convex hull) and finds the triangle pierced by the ray pointing
        towards the query direction. Returns ``(indices, weights)`` or
        ``None`` if the direction lies outside the measured region or the
        hull is degenerate (e.g. coplanar measurements).
        """
        from scipy.spatial import ConvexHull, QhullError

        if self._hull_cache is None:
            try:
                self._hull_cache = ConvexHull(self._unit_vectors)
            except QhullError:
                self._hull_cache = False  # mark as unavailable
        if self._hull_cache is False:
            return None

        tol = 1e-9
        for simplex in self._hull_cache.simplices:
            verts = self._unit_vectors[simplex]  # (3, 3)
            # Solve verts.T @ w = query for barycentric weights w.
            try:
                w = np.linalg.solve(verts.T, query)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate face
                continue
            if np.all(w >= -tol):
                w = np.clip(w, 0, None)
                return np.asarray(simplex), w / w.sum()
        return None

    def _ring_weights(self, query):
        """Bracketing-pair indices/weights for a planar measurement ring.

        Used when the measured directions are coplanar (e.g. a horizontal
        ring). The query is projected onto the ring plane and linearly
        interpolated between the two angularly adjacent measurements. Any
        out-of-plane component of the query (e.g. an elevation request on a
        horizontal-only set) is dropped by the projection.
        """
        vecs = self._unit_vectors
        centered = vecs - vecs.mean(0)
        # plane normal = direction of least variance
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]

        # project every direction (and the query) into the plane through the
        # origin parallel to the ring, then renormalise to unit length
        def _project(v):
            p = v - np.outer(v @ normal, normal) if v.ndim > 1 else v - (v @ normal) * normal
            norm = np.linalg.norm(p, axis=-1, keepdims=v.ndim > 1)
            return p / norm

        proj = _project(vecs)
        qp = _project(query[None])[0]

        # 2-D basis within the plane
        e1 = proj[0]
        e2 = np.cross(normal, e1)
        angles = np.arctan2(proj @ e2, proj @ e1)
        qa = np.arctan2(qp @ e2, qp @ e1)

        order = np.argsort(angles)
        sorted_ang = angles[order]
        # cyclic search for the bracketing pair
        n = len(order)
        for i in range(n):
            a0 = sorted_ang[i]
            a1 = sorted_ang[(i + 1) % n]
            span = (a1 - a0) % (2 * np.pi)
            offset = (qa - a0) % (2 * np.pi)
            if offset <= span + 1e-12 and span > 0:
                frac = offset / span
                idx = np.array([order[i], order[(i + 1) % n]])
                return idx, np.array([1 - frac, frac])
        # numerical fallback: snap to nearest
        best = int(np.argmax(vecs @ query))
        return np.array([best]), np.array([1.0])

    def _interp_weights(self, azimuth, elevation):
        """Direction indices and weights used to interpolate an HRIR."""
        query = _spherical_to_cartesian(azimuth, elevation)[0]
        dots = self._unit_vectors @ query
        best = int(np.argmax(dots))

        # exact (or essentially exact) hit on a measured direction
        if dots[best] > 1 - 1e-12 or self.n_directions < 3:
            return np.array([best]), np.array([1.0])

        # intrinsic dimensionality of the measurement grid
        centered = self._unit_vectors - self._unit_vectors.mean(0)
        rank = np.linalg.matrix_rank(centered, tol=1e-7)

        if rank >= 3:
            result = self._barycentric_weights_3d(query)
            if result is not None:
                return result
            return np.array([best]), np.array([1.0])
        if rank == 2:
            return self._ring_weights(query)
        # colinear / single point
        return np.array([best]), np.array([1.0])

    def interpolate(self, azimuth: float, elevation: float) -> Signal:
        """Return an interpolated HRIR for an arbitrary direction.

        For a fully three-dimensional measurement grid the impulse responses
        of the three directions forming the surrounding spherical triangle
        are combined using barycentric weights. For a planar grid (e.g. a
        horizontal ring) the two angularly adjacent directions are
        interpolated instead. If the requested direction lies outside the
        measured region, the nearest measured HRIR is returned.

        Note that this performs a straight linear combination of the impulse
        responses; for widely spaced measurements time-alignment before
        interpolation would reduce comb-filtering artefacts.

        Parameters
        ----------
        azimuth : float
            Query azimuth in degrees.
        elevation : float
            Query elevation in degrees.

        Returns
        -------
        Signal
            A two-channel (left, right) signal of shape ``(n_taps, 2)``.
        """
        indices, weights = self._interp_weights(azimuth, elevation)
        out = Signal(2, self.n_taps / self.fs, self.fs, dtype=self.hrirs.dtype)
        for idx, weight in zip(indices, weights):
            out += self.hrirs.ch[idx] * weight
        return out

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(
        self,
        signal,
        azimuth: float,
        elevation: float,
        interpolate: bool = True,
        mode: str = "full",
    ) -> Signal:
        """Spatialize a mono signal to a binaural (two-channel) signal.

        Convolves the input with the (left, right) HRIR for the requested
        direction. The input is left untouched; a new two-channel
        :class:`audiotoolbox.Signal` is returned.

        Parameters
        ----------
        signal : Signal or ndarray
            The mono source signal. If an ndarray is passed it is wrapped
            using the HRIR set's sampling rate.
        azimuth : float
            Source azimuth in degrees.
        elevation : float
            Source elevation in degrees.
        interpolate : bool, optional
            If True (default) the HRIR is barycentrically interpolated,
            otherwise the nearest measured HRIR is used.
        mode : {'full', 'same', 'valid'}, optional
            Convolution mode passed to :meth:`audiotoolbox.Signal.convolve`
            (default = ``'full'``).

        Returns
        -------
        Signal
            The binaural output of shape ``(n_out, 2)``.
        """
        src = audio.as_signal(signal, self.fs)
        if src.fs != self.fs:
            raise ValueError(
                f"Sampling rate mismatch: signal at {src.fs} Hz, HRIRs at "
                f"{self.fs} Hz."
            )
        if src.n_channels != 1:
            raise ValueError("render expects a mono signal.")

        hrir = (
            self.interpolate(azimuth, elevation)
            if interpolate
            else self.nearest(azimuth, elevation)
        )

        # convolve mutates and resizes in place, so work on a copy to leave
        # the caller's signal untouched.
        out = src.copy()
        out.convolve(hrir, mode=mode)
        return out

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Short human-readable description of the HRIR set."""
        return (
            f"{self.n_directions} directions @ {self.fs} Hz | "
            f"{self.n_taps} taps | coords: {self.coordinate_system}"
        )

    def __repr__(self) -> str:
        return f"<HRIRSet: {self.summary()}>"


__all__ = ["HRIRSet"]
