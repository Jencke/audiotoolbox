from __future__ import annotations

from typing import Literal, Optional

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
    :class:`audiotoolbox.Signal` of shape ``(n_taps, 2, n_directions)`` where
    the first channel axis holds the (left, right) ear. The associated source
    directions are kept in a separate position table.

    Parameters
    ----------
    hrirs : Signal
        Time-domain impulse responses of shape ``(n_taps, 2, n_directions)``.
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
    >>> hrirs = audio.Signal((2, 4), 128 / 48000, 48000)  # L/R x 4 directions
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
        if len(channel_shape) != 2 or channel_shape[0] != 2:
            raise ValueError(
                "hrirs must be a Signal of shape (n_taps, 2, n_directions); "
                f"got channel shape {channel_shape}."
            )
        if channel_shape[1] != len(positions):
            raise ValueError(
                f"Number of directions does not match: {channel_shape[1]} "
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

        hrirs = Signal((2, n_directions), n_taps / fs, fs)
        # reorder to (n_taps, 2, n_directions)
        hrirs[:] = np.transpose(ir, (2, 1, 0))

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
        return self.hrirs.channel_shape[1]

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
        ``(n_taps, 2, n_directions)`` is returned.

        Returns
        -------
        FrequencyDomainSignal
            The head-related transfer functions.
        """
        return self.hrirs.to_freqdomain()

    def phase_shifts(self, unwrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return HRTF phase for all ears and directions.

        The phase is computed from the frequency-domain transfer functions of
        the stored HRIRs. The returned phase array follows the internal HRTF
        layout ``(n_freq, 2, n_directions)`` where the first channel axis is
        ``(left, right)`` and the last axis enumerates source directions.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase along the frequency axis before
            returning it (default = ``False``).

        Returns
        -------
        tuple of ndarray
            ``(frequency, phase)`` where ``frequency`` is in Hz and ``phase``
            is in radians.
        """
        hrtf = self.to_hrtf()
        phase = hrtf.phase
        if unwrap:
            phase = np.unwrap(phase, axis=0)
        return hrtf.freq, phase

    def get_ipd(
        self, unwrap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return interaural phase differences for all directions.

        The interaural phase difference (IPD) is defined here as left-ear
        phase minus right-ear phase, yielding an array of shape
        ``(n_freq, n_directions)``.

        .. math::

            \Delta\varphi = \varphi_\mathrm{left} - \varphi_\mathrm{right}

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the per-ear phases along the frequency axis
            before taking the difference (default = ``False``).

        Returns
        -------
        tuple of ndarray
            ``(frequency, ipd)`` where ``frequency`` is in Hz and ``ipd`` is
            in radians.
        """
        freq, phase = self.phase_shifts(unwrap=unwrap)
        ipd = phase[:, 0, :] - phase[:, 1, :]
        ipd = ipd[freq >= 0]
        freq = freq[freq >= 0]
        return freq, ipd

    def get_ild(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return interaural level differences for all directions.

        The interaural level difference (ILD) is defined here as left-ear
        level minus right-ear level, computed from the HRTF magnitudes as
        
        .. math::

            \Delta L = 20 \log_{10}\!\left(\frac{|H_\mathrm{left}|}{|H_\mathrm{right}|}\right)

        Returns
        -------
        tuple of ndarray
            ``(frequency, ild)`` where ``frequency`` is in Hz and ``ild`` is
            in dB, with shape ``(n_freq, n_directions)``.
        """
        hrtf = self.to_hrtf()
        mag = np.asarray(hrtf.mag)
        floor = np.finfo(mag.dtype).tiny
        ild = 20 * np.log10(np.maximum(mag[:, 0, :], floor) / np.maximum(mag[:, 1, :], floor))
        ild = ild[hrtf.freq >= 0]
        freq = hrtf.freq[hrtf.freq >= 0]
        return freq, ild
    
    def get_itd(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return interaural time differences for all directions.

        The interaural time difference (ITD) is computed from the unwrapped
        interaural phase difference (IPD) as

        .. math::

            \Delta t = \frac{\Delta\varphi}{2\pi f}

        The returned ITD is in seconds and has shape ``(n_freq, n_directions)``.

        Returns
        -------
        tuple of ndarray
            ``(frequency, itd)`` where ``frequency`` is in Hz and ``itd``
            is in seconds, with shape ``(n_freq, n_directions)``.
        """
        freq, ipd = self.get_ipd(unwrap=True)
        itd = ipd / (2 * np.pi * freq[:, np.newaxis])
        itd = itd[freq >= 0]
        freq = freq[freq >= 0]
        return freq, itd

    def group_delays(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Return per-ear group delay for all directions.

        Group delay is computed from the slope of the unwrapped HRTF phase
        along the frequency-bin axis:

        .. math::

            t_g = -\frac{\mathrm{d}\phi}{\mathrm{d}\omega}

        The returned array has shape ``(n_freq, 2, n_directions)`` and is in
        seconds.

        Returns
        -------
        tuple of ndarray
            ``(frequency, delay)`` where ``frequency`` is in Hz and
            ``delay`` is in seconds.
        """
        freq, phase = self.phase_shifts(unwrap=True)
        delta_f = self.fs / self.n_taps
        delay = -np.gradient(phase, delta_f, axis=0) / (2 * np.pi)
        delay = delay[freq >= 0]
        freq = freq[freq >= 0]
        return freq, delay

    # ------------------------------------------------------------------
    # spatial lookup
    # ------------------------------------------------------------------
    def _nearest_index(self, azimuth: float, elevation: float) -> int:
        """Index of the measured direction closest to the query direction."""
        query = _spherical_to_cartesian(azimuth, elevation)[0]
        # Nearest on the unit sphere == largest dot product (cosine distance).
        dots = self._unit_vectors @ query
        return int(np.argmax(dots))

    def _subset(self, indices) -> "HRIRSet":
        """Return a directional subset as a new HRIRSet."""
        idx = np.asarray(indices, dtype=int).ravel()
        sub = Signal((2, len(idx)), self.n_taps / self.fs, self.fs, dtype=self.hrirs.dtype)
        sub[:] = np.asarray(self.hrirs)[:, :, idx]
        return HRIRSet(
            sub,
            self.positions[idx],
            coordinate_system=self.coordinate_system,
            units=self.units,
        )

    def nearest(self, azimuth, elevation) -> "HRIRSet":
        """Return the HRIR of the closest measured direction.

        Parameters
        ----------
        azimuth : float or array_like
            Query azimuth(s) in degrees.
        elevation : float or array_like
            Query elevation(s) in degrees.

        Returns
        -------
        HRIRSet
            Subset of one (scalar query) or many (vectorized query)
            nearest measured directions.
        """
        az = np.asarray(azimuth, dtype=float)
        el = np.asarray(elevation, dtype=float)
        az, el = np.broadcast_arrays(az, el)

        if az.ndim == 0:
            idx = self._nearest_index(float(az), float(el))
            return self._subset([idx])

        idx = [self._nearest_index(float(a), float(e)) for a, e in zip(az.ravel(), el.ravel())]
        return self._subset(idx)

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

    def interpolate(self, azimuth: float, elevation: float) -> "HRIRSet":
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
        HRIRSet
            Single-direction HRIRSet containing the interpolated HRIR.
        """
        indices, weights = self._interp_weights(azimuth, elevation)
        out = Signal((2, 1), self.n_taps / self.fs, self.fs, dtype=self.hrirs.dtype)
        for idx, weight in zip(indices, weights):
            out.ch[:, 0] += self.hrirs.ch[:, idx] * weight
        pos = np.array([azimuth, elevation], dtype=float)
        return HRIRSet(
            out,
            pos[None, :],
            coordinate_system=self.coordinate_system,
            units=self.units,
        )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def render(
        self,
        signal,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        interpolate: bool = True,
        mode: Literal["full", "same", "valid"] = "full",
    ) -> Signal:
        """Spatialize a signal with one or many HRIR directions.

        If ``azimuth``/``elevation`` are provided, a single direction is
        selected (nearest or interpolated) and rendered as binaural output.

        If no direction is provided, this HRIRSet is used directly:
        - mono input ``(n, 1)`` is rendered against all stored directions
        - multichannel input ``(n, K)`` must match ``K == n_directions`` and
          is rendered channel-wise against matching HRIR directions.

        Batched set rendering returns output with channel shape ``(2, K)``
        i.e. array shape ``(n_out, 2, K)``.

        Parameters
        ----------
        signal : Signal or ndarray
            The mono source signal. If an ndarray is passed it is wrapped
            using the HRIR set's sampling rate.
        azimuth : float, optional
            Source azimuth in degrees.
        elevation : float, optional
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
            Binaural output. Single-direction rendering returns shape
            ``(n_out, 2)``; batched set rendering returns ``(n_out, 2, K)``.
        """
        src = audio.as_signal(signal, self.fs)
        if src.fs != self.fs:
            raise ValueError(
                f"Sampling rate mismatch: signal at {src.fs} Hz, HRIRs at "
                f"{self.fs} Hz."
            )

        if (azimuth is None) != (elevation is None):
            raise ValueError("azimuth and elevation must either both be set or both be omitted")

        if azimuth is not None and elevation is not None:
            subset = (
                self.interpolate(azimuth, elevation)
                if interpolate
                else self.nearest(azimuth, elevation)
            )
            out = subset.render(src, mode=mode)
            if out.n_channels == (2, 1):
                return audio.as_signal(np.asarray(out)[:, :, 0], self.fs)
            return out

        if src.n_channels not in (1, self.n_directions):
            raise ValueError(
                "set rendering expects mono input or matching channel count; "
                f"got {src.n_channels} channels for {self.n_directions} HRIR directions"
            )

        out = src.copy()
        out.convolve(self.hrirs, mode=mode)
        if self.n_directions == 1 and out.n_channels == 2:
            return audio.as_signal(np.asarray(out)[:, :, np.newaxis], self.fs)
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
