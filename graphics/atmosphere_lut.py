"""Altitude-aware, deterministic atmospheric-scattering LUT.

The module models the parts of a sky precomputation pipeline that are easy to
get wrong:

* scattering coefficients are recomputed for the current altitude on every
  precomputation (``altitude_recalc``);
* the sun-elevation axis is densified around the horizon and sampled with a
  C1 smoothstep interpolator, and the horizon rows are low-pass filtered, so
  the sunrise boundary no longer shows colour banding (``lut_smooth``);
* an ordered write fence publishes the finished LUT by flipping a generation
  counter under a condition variable, so the skybox pass never observes a
  partially written table (``write_fence`` / ``skybox_bind``);
* turbidity is part of the table key, so a weather change drops every stale
  entry instead of sampling it (``turbidity_invalidate`` / ``weather_switch``);
* the precomputation key is a canonical BLAKE2b hash of the normalised
  parameters, independent of dict order, hash randomisation and clock
  (``param_hash``);
* sampling quantises its inputs and outputs, and precomputation has no hidden
  state, so two runs at the same sun angle are bit-identical
  (``sample_determinism`` / ``determinism``).

Every public helper accepts a scalar, a mapping of aliases, an
``AtmosphereParams`` instance or an already built ``AtmosphereLUT``/result
mapping, so the module is usable both from a renderer and from tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import threading
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import numpy as np

__all__ = [
    "AtmosphereParams",
    "AtmosphereLUT",
    "LUTCache",
    "WriteFence",
    "altitude_recalc",
    "lut_smooth",
    "write_fence",
    "turbidity_invalidate",
    "param_hash",
    "sample_determinism",
    "sun_angle",
    "validation",
    "audit",
    "fallback",
    "precompute",
    "metrics",
    "weather_switch",
    "skybox_bind",
    "determinism",
    "sample",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAYLEIGH_SCALE_HEIGHT_M = 8_000.0
MIE_SCALE_HEIGHT_M = 1_200.0

# Sea-level Rayleigh scattering coefficients (1/m) for roughly 680/550/440 nm.
BETA_RAYLEIGH_0 = (5.802e-6, 13.558e-6, 33.100e-6)
BETA_MIE_0 = 21.0e-6
MIE_EXTINCTION_RATIO = 1.10
MIE_G = 0.76

HORIZON_MIN_ELEVATION_DEG = -8.0
HORIZON_MAX_ELEVATION_DEG = 90.0
HORIZON_BAND_DEG = 15.0
HORIZON_SIGMA_DEG = 4.0
HORIZON_WARP_POWER = 1.7

DEFAULT_SUN_SAMPLES = 32
DEFAULT_VIEW_SAMPLES = 16
MIN_SAMPLES = 4
MAX_SAMPLES = 512

EXPOSURE = 1.5
SAMPLE_QUANTUM = 1e-7
INPUT_QUANTUM = 1e-6
HASH_QUANTUM = 1e-9
EPS = 1e-12

HASH_VERSION = "atmosphere-lut-v1"

_FIELD_ALIASES = {
    "altitude_m": "altitude_m",
    "altitude": "altitude_m",
    "height_m": "altitude_m",
    "height": "altitude_m",
    "elevation_m": "altitude_m",
    "turbidity": "turbidity",
    "haze": "turbidity",
    "turbidity_scale": "turbidity",
    "sun_elevation_deg": "sun_elevation_deg",
    "sun_elevation": "sun_elevation_deg",
    "sun_angle": "sun_elevation_deg",
    "sun_angle_deg": "sun_elevation_deg",
    "elevation_deg": "sun_elevation_deg",
    "sun_azimuth_deg": "sun_azimuth_deg",
    "sun_azimuth": "sun_azimuth_deg",
    "azimuth_deg": "sun_azimuth_deg",
    "azimuth": "sun_azimuth_deg",
    "lut_sun_samples": "lut_sun_samples",
    "sun_samples": "lut_sun_samples",
    "lut_view_samples": "lut_view_samples",
    "view_samples": "lut_view_samples",
}

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphereParams:
    """Everything that influences the precomputed table."""

    altitude_m: float = 0.0
    turbidity: float = 2.0
    sun_elevation_deg: float = 10.0
    sun_azimuth_deg: float = 0.0
    lut_sun_samples: int = DEFAULT_SUN_SAMPLES
    lut_view_samples: int = DEFAULT_VIEW_SAMPLES


_PARAM_NAMES = [f.name for f in fields(AtmosphereParams)]


def _convert_field(name: str, raw: Any) -> Any:
    if name in ("lut_sun_samples", "lut_view_samples"):
        return int(round(float(raw)))
    return float(raw)


def _coerce_params(value: Any = None, scalar_field: str = "altitude_m") -> AtmosphereParams:
    """Normalise a scalar, mapping, sequence or params object to params."""

    if isinstance(value, AtmosphereParams):
        return value
    if value is None:
        return AtmosphereParams()
    if isinstance(value, Mapping):
        kwargs: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if raw_value is None:
                continue
            key = str(raw_key).strip().lower()
            if key == "altitude_km":
                kwargs["altitude_m"] = float(raw_value) * 1000.0
                continue
            name = _FIELD_ALIASES.get(key)
            if name is None:
                continue
            kwargs[name] = _convert_field(name, raw_value)
        return AtmosphereParams(**kwargs)
    if isinstance(value, (list, tuple)):
        positional = ["altitude_m", "turbidity", "sun_elevation_deg", "sun_azimuth_deg"]
        kwargs = {}
        for name, item in zip(positional, value):
            if item is not None:
                kwargs[name] = _convert_field(name, item)
        return AtmosphereParams(**kwargs)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return AtmosphereParams(**{scalar_field: _convert_field(scalar_field, value)})
    params = getattr(value, "params", None)
    if isinstance(params, AtmosphereParams):
        return params
    raise TypeError(f"cannot interpret {type(value)!r} as atmosphere parameters")


def _apply_overrides(params: AtmosphereParams, overrides: Mapping[str, Any]) -> AtmosphereParams:
    if not overrides:
        return params
    kwargs: dict[str, Any] = {}
    for raw_key, raw_value in overrides.items():
        if raw_value is None:
            continue
        key = str(raw_key).strip().lower()
        if key == "altitude_km":
            kwargs["altitude_m"] = float(raw_value) * 1000.0
            continue
        name = _FIELD_ALIASES.get(key)
        if name is None:
            continue
        kwargs[name] = _convert_field(name, raw_value)
    return replace(params, **kwargs) if kwargs else params


def _params_dict(params: AtmosphereParams) -> dict[str, Any]:
    return {name: getattr(params, name) for name in _PARAM_NAMES}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _param_errors(params: AtmosphereParams) -> list[str]:
    errors: list[str] = []
    if not math.isfinite(params.altitude_m):
        errors.append("altitude_m must be finite")
    elif not (-500.0 <= params.altitude_m <= 100_000.0):
        errors.append("altitude_m outside [-500, 100000]")
    if not math.isfinite(params.turbidity):
        errors.append("turbidity must be finite")
    elif not (0.0 <= params.turbidity <= 20.0):
        errors.append("turbidity outside [0, 20]")
    if not math.isfinite(params.sun_elevation_deg):
        errors.append("sun_elevation_deg must be finite")
    elif not (-90.0 <= params.sun_elevation_deg <= 90.0):
        errors.append("sun_elevation_deg outside [-90, 90]")
    if not math.isfinite(params.sun_azimuth_deg):
        errors.append("sun_azimuth_deg must be finite")
    elif not (-360.0 <= params.sun_azimuth_deg <= 360.0):
        errors.append("sun_azimuth_deg outside [-360, 360]")
    if not (MIN_SAMPLES <= params.lut_sun_samples <= MAX_SAMPLES):
        errors.append(f"lut_sun_samples outside [{MIN_SAMPLES}, {MAX_SAMPLES}]")
    if not (MIN_SAMPLES <= params.lut_view_samples <= MAX_SAMPLES):
        errors.append(f"lut_view_samples outside [{MIN_SAMPLES}, {MAX_SAMPLES}]")
    return errors


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def _sun_direction(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    elevation = math.radians(elevation_deg)
    azimuth = math.radians(azimuth_deg)
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def _airmass(elevation_deg: float) -> float:
    """Kasten-Young relative air mass, clamped to the twilight range."""

    elevation = _clamp(float(elevation_deg), -2.0, 90.0)
    denominator = math.sin(math.radians(elevation)) + 0.50572 * max(
        elevation + 6.07995, 1e-3
    ) ** -1.6364
    return 1.0 / max(denominator, 1e-3)


def _coefficients(params: AtmosphereParams) -> dict[str, Any]:
    """Recompute scattering coefficients at the *current* altitude."""

    altitude = max(float(params.altitude_m), 0.0)
    pressure_ratio = math.exp(-altitude / RAYLEIGH_SCALE_HEIGHT_M)
    density_ratio = math.exp(-altitude / MIE_SCALE_HEIGHT_M)
    rayleigh = tuple(beta * pressure_ratio for beta in BETA_RAYLEIGH_0)
    mie = BETA_MIE_0 * max(float(params.turbidity), 0.0) * density_ratio
    mie_extinction = mie * MIE_EXTINCTION_RATIO
    return {
        "rayleigh": rayleigh,
        "mie": mie,
        "mie_extinction": mie_extinction,
        "pressure_ratio": pressure_ratio,
        "density_ratio": density_ratio,
    }


def _optical_depth(params: AtmosphereParams) -> tuple[float, float, float]:
    coeff = _coefficients(params)
    rayleigh = coeff["rayleigh"]
    mie_extinction = coeff["mie_extinction"]
    return tuple(
        rayleigh[i] * RAYLEIGH_SCALE_HEIGHT_M + mie_extinction * MIE_SCALE_HEIGHT_M
        for i in range(3)
    )


def _sun_transmittance(params: AtmosphereParams, elevation_deg: float) -> tuple[float, float, float]:
    airmass = _airmass(elevation_deg)
    return tuple(math.exp(-tau * airmass) for tau in _optical_depth(params))


def _rayleigh_phase(cos_theta: float) -> float:
    return 3.0 / (16.0 * math.pi) * (1.0 + cos_theta * cos_theta)


def _mie_phase(cos_theta: float) -> float:
    g = MIE_G
    denominator = max(1.0 + g * g - 2.0 * g * cos_theta, 1e-6)
    return (1.0 - g * g) / (4.0 * math.pi * denominator ** 1.5)


def _sky_radiance(
    params: AtmosphereParams,
    sun_elevation_deg: float,
    view_zenith_deg: float,
) -> tuple[float, float, float]:
    """Single-scattering airlight for one sun/view direction pair."""

    coeff = _coefficients(params)
    beta_rayleigh = np.asarray(coeff["rayleigh"], dtype=np.float64)
    beta_mie = float(coeff["mie"])
    mie_extinction = float(coeff["mie_extinction"])

    tau_sun = (
        beta_rayleigh * RAYLEIGH_SCALE_HEIGHT_M
        + mie_extinction * MIE_SCALE_HEIGHT_M
    ) * _airmass(sun_elevation_deg)
    sun_transmittance = np.exp(-tau_sun)

    view_zenith = math.radians(_clamp(view_zenith_deg, 0.0, 89.5))
    cos_view = max(math.cos(view_zenith), 0.01)
    tau_view = (
        beta_rayleigh * RAYLEIGH_SCALE_HEIGHT_M
        + mie_extinction * MIE_SCALE_HEIGHT_M
    ) / cos_view
    view_transmittance = 1.0 - np.exp(-tau_view)

    sun = _sun_direction(sun_elevation_deg, params.sun_azimuth_deg)
    view = (math.sin(view_zenith), 0.0, math.cos(view_zenith))
    cos_theta = _clamp(sun[0] * view[0] + sun[2] * view[2], -1.0, 1.0)

    phase_rayleigh = _rayleigh_phase(cos_theta)
    phase_mie = _mie_phase(cos_theta)
    scattering = beta_rayleigh * phase_rayleigh + beta_mie * phase_mie
    extinction = beta_rayleigh + mie_extinction + EPS

    airlight = (scattering / extinction) * sun_transmittance * view_transmittance
    return tuple(float(EXPOSURE * channel) for channel in airlight)


# ---------------------------------------------------------------------------
# LUT storage, grids and sampling
# ---------------------------------------------------------------------------


def _sun_elevation_nodes(count: int) -> np.ndarray:
    """Horizon-dense sun-elevation axis (warped, monotonic)."""

    u = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    return HORIZON_MIN_ELEVATION_DEG + (
        HORIZON_MAX_ELEVATION_DEG - HORIZON_MIN_ELEVATION_DEG
    ) * np.power(u, HORIZON_WARP_POWER)


def _view_zenith_nodes(count: int) -> np.ndarray:
    u = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    return u * 90.0


def _checksum(data: np.ndarray) -> str:
    digest = hashlib.blake2b(np.ascontiguousarray(data).tobytes(), digest_size=16)
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
class AtmosphereLUT:
    """Immutable table ``[sun elevation, view zenith, rgb]``."""

    key: str
    params: AtmosphereParams
    sun_elevations_deg: np.ndarray
    view_zeniths_deg: np.ndarray
    data: np.ndarray
    smoothed: bool = False
    checksum: str = ""

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AtmosphereLUT)
            and self.key == other.key
            and self.checksum == other.checksum
            and self.smoothed == other.smoothed
        )

    def __hash__(self) -> int:
        return hash((self.key, self.checksum, self.smoothed))

    def sample(
        self,
        sun_elevation_deg: float,
        view_zenith_deg: float,
        *,
        smooth: bool = True,
        quantize: bool = True,
    ) -> tuple[float, float, float]:
        return _sample_lut(
            self,
            sun_elevation_deg,
            view_zenith_deg,
            smooth=smooth,
            quantize=quantize,
        )


def _axis_weights(axis: np.ndarray, x: float) -> tuple[int, int, float]:
    count = int(axis.size)
    if count == 1:
        return 0, 0, 0.0
    if x <= axis[0]:
        return 0, 0, 0.0
    if x >= axis[-1]:
        return count - 1, count - 1, 0.0
    upper = int(np.searchsorted(axis, x, side="right"))
    i0 = min(max(upper - 1, 0), count - 2)
    i1 = i0 + 1
    span = float(axis[i1] - axis[i0])
    if span <= 0.0:
        return i0, i0, 0.0
    t = (x - float(axis[i0])) / span
    return i0, i1, _clamp(t, 0.0, 1.0)


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _quantize(value: float, quantum: float = SAMPLE_QUANTUM) -> float:
    rounded = round(float(value) / quantum) * quantum
    return 0.0 if rounded == 0.0 else float(rounded)


def _sample_lut(
    lut: AtmosphereLUT,
    sun_elevation_deg: float,
    view_zenith_deg: float,
    *,
    smooth: bool = True,
    quantize: bool = True,
) -> tuple[float, float, float]:
    data = np.asarray(lut.data, dtype=np.float64)
    i0, i1, t = _axis_weights(np.asarray(lut.sun_elevations_deg, dtype=np.float64), float(sun_elevation_deg))
    j0, j1, u = _axis_weights(np.asarray(lut.view_zeniths_deg, dtype=np.float64), float(view_zenith_deg))
    if smooth:
        t = _smoothstep(t)
        u = _smoothstep(u)

    value = (
        data[i0, j0] * ((1.0 - t) * (1.0 - u))
        + data[i1, j0] * (t * (1.0 - u))
        + data[i0, j1] * ((1.0 - t) * u)
        + data[i1, j1] * (t * u)
    )
    if quantize:
        return tuple(_quantize(channel) for channel in value)
    return tuple(float(channel) for channel in value)


def _build_lut(params: AtmosphereParams) -> AtmosphereLUT:
    sun_count = int(_clamp(params.lut_sun_samples, MIN_SAMPLES, MAX_SAMPLES))
    view_count = int(_clamp(params.lut_view_samples, MIN_SAMPLES, MAX_SAMPLES))
    sun_nodes = _sun_elevation_nodes(sun_count)
    view_nodes = _view_zenith_nodes(view_count)

    data = np.empty((sun_count, view_count, 3), dtype=np.float64)
    for i, elevation in enumerate(sun_nodes):
        for j, zenith in enumerate(view_nodes):
            data[i, j] = _sky_radiance(params, float(elevation), float(zenith))
    np.maximum(data, 0.0, out=data)
    data32 = np.ascontiguousarray(data.astype(np.float32))

    return AtmosphereLUT(
        key=param_hash(params),
        params=params,
        sun_elevations_deg=sun_nodes,
        view_zeniths_deg=view_nodes,
        data=data32,
        smoothed=False,
        checksum=_checksum(data32),
    )


# ---------------------------------------------------------------------------
# Horizon smoothing
# ---------------------------------------------------------------------------


def _banding_metric(data: np.ndarray, elevations: np.ndarray, band_deg: float = HORIZON_BAND_DEG) -> float:
    """Largest second difference of RGB along the sun-elevation axis."""

    array = np.asarray(data, dtype=np.float64)
    if array.shape[0] < 3:
        return 0.0
    second = array[2:] - 2.0 * array[1:-1] + array[:-2]
    mask = np.abs(elevations[1:-1]) <= band_deg
    if not bool(np.any(mask)):
        mask = np.ones(elevations.shape[0] - 2, dtype=bool)
    return float(np.max(np.abs(second[mask])))


def _gradient_metric(data: np.ndarray, elevations: np.ndarray, band_deg: float = HORIZON_BAND_DEG) -> float:
    array = np.asarray(data, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    delta = array[1:] - array[:-1]
    mask = np.abs(elevations[1:]) <= band_deg
    if not bool(np.any(mask)):
        mask = np.ones(elevations.shape[0] - 1, dtype=bool)
    return float(np.max(np.abs(delta[mask])))


def _smooth_lut_data(
    data: np.ndarray,
    elevations: np.ndarray,
    sigma_deg: float = HORIZON_SIGMA_DEG,
    band_deg: float = HORIZON_BAND_DEG,
    passes: int = 2,
) -> np.ndarray:
    """Gaussian-blur the horizon rows of the sun-elevation axis.

    The blur weight fades out away from the horizon, so the rest of the table
    keeps its physical values while the steep sunrise transition is widened
    enough that trilinear lookups cannot alias into bands.
    """

    array = np.asarray(data, dtype=np.float64)
    if array.shape[0] < 3 or passes <= 0:
        return array.copy()

    distance = elevations[:, None] - elevations[None, :]
    kernel = np.exp(-0.5 * (distance / max(sigma_deg, 1e-6)) ** 2)
    kernel /= np.sum(kernel, axis=1, keepdims=True)
    weight = np.exp(-0.5 * (elevations / max(band_deg, 1e-6)) ** 2)
    weight = np.clip(weight, 0.0, 1.0)[:, None, None]

    smoothed = array
    for _ in range(int(passes)):
        blurred = np.einsum("ij,jkc->ikc", kernel, smoothed)
        smoothed = smoothed * (1.0 - weight) + blurred * weight
    return smoothed


def _with_horizon_smoothing(
    lut: AtmosphereLUT,
    sigma_deg: float = HORIZON_SIGMA_DEG,
    band_deg: float = HORIZON_BAND_DEG,
    passes: int = 2,
) -> AtmosphereLUT:
    smoothed = _smooth_lut_data(
        lut.data,
        np.asarray(lut.sun_elevations_deg, dtype=np.float64),
        sigma_deg=sigma_deg,
        band_deg=band_deg,
        passes=passes,
    )
    data32 = np.ascontiguousarray(smoothed.astype(np.float32))
    return replace(lut, data=data32, smoothed=True, checksum=_checksum(data32))


def _as_lut(value: Any) -> AtmosphereLUT:
    if isinstance(value, AtmosphereLUT):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("lut")
        if isinstance(candidate, AtmosphereLUT):
            return candidate
    return _build_lut(_coerce_params(value))


# ---------------------------------------------------------------------------
# Write fence: staging -> commit -> published snapshot
# ---------------------------------------------------------------------------


class WriteFence:
    """Ordered publication of an asynchronously built LUT.

    Writers ``stage`` a fully built, immutable ``AtmosphereLUT`` and then
    ``commit`` it.  Readers only ever see a published snapshot plus its
    generation number; a reader that needs a newer generation can ``wait`` for
    it.  Committing installs the new reference in one assignment, so the
    skybox draw can never observe a half-written table.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._generation = 0
        self._published: AtmosphereLUT | None = None
        self._staged: AtmosphereLUT | None = None
        self._commits = 0
        self._drops = 0

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def stage(self, lut: AtmosphereLUT) -> int:
        if not isinstance(lut, AtmosphereLUT):
            raise TypeError("stage() expects an AtmosphereLUT")
        with self._condition:
            self._staged = lut
            return self._generation

    def commit(self, lut: AtmosphereLUT | None = None) -> int:
        with self._condition:
            candidate = lut if lut is not None else self._staged
            if candidate is None:
                raise RuntimeError("commit() without a staged LUT")
            self._published = candidate
            self._staged = None
            self._generation += 1
            self._commits += 1
            self._condition.notify_all()
            return self._generation

    def wait(self, generation: int, timeout: float = 2.0) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._generation >= int(generation), timeout=timeout
            )

    def read(self) -> dict[str, Any]:
        with self._condition:
            snapshot = self._published
            return {
                "generation": self._generation,
                "published": snapshot is not None,
                "key": None if snapshot is None else snapshot.key,
                "checksum": None if snapshot is None else snapshot.checksum,
                "params": None if snapshot is None else _params_dict(snapshot.params),
            }

    def snapshot(self) -> AtmosphereLUT | None:
        with self._condition:
            return self._published

    def invalidate(self, reason: str = "manual") -> int:
        with self._condition:
            had = self._published is not None
            self._published = None
            self._staged = None
            self._generation += 1
            if had:
                self._drops += 1
            return self._generation

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "generation": self._generation,
                "commits": self._commits,
                "drops": self._drops,
            }


# ---------------------------------------------------------------------------
# Cache keyed by the canonical parameter hash
# ---------------------------------------------------------------------------


class LUTCache:
    """Small insertion-ordered cache; the key always includes turbidity."""

    def __init__(self, capacity: int = 16) -> None:
        self._lock = threading.RLock()
        self._capacity = max(int(capacity), 1)
        self._entries: dict[str, AtmosphereLUT] = {}
        self._order: list[str] = []

    def get(self, params: AtmosphereParams) -> AtmosphereLUT | None:
        key = param_hash(params)
        with self._lock:
            return self._entries.get(key)

    def store(self, lut: AtmosphereLUT) -> str:
        if not isinstance(lut, AtmosphereLUT):
            raise TypeError("store() expects an AtmosphereLUT")
        with self._lock:
            key = lut.key
            if key in self._entries:
                self._order.remove(key)
            self._entries[key] = lut
            self._order.append(key)
            while len(self._order) > self._capacity:
                evicted = self._order.pop(0)
                self._entries.pop(evicted, None)
            return key

    def get_or_build(self, params: AtmosphereParams) -> AtmosphereLUT:
        with self._lock:
            cached = self._entries.get(param_hash(params))
            if cached is not None:
                return cached
            lut = _build_lut(params)
            self.store(lut)
            return lut

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._order)

    def entries(self) -> dict[str, AtmosphereLUT]:
        with self._lock:
            return dict(self._entries)

    def stale_keys(self, turbidity: float) -> list[str]:
        target = _canonical_value(turbidity)
        with self._lock:
            return [
                key
                for key, lut in self._entries.items()
                if _canonical_value(lut.params.turbidity) == target
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def invalidate_turbidity(self, turbidity: float) -> list[str]:
        target = _canonical_value(turbidity)
        with self._lock:
            removed = [
                key
                for key, lut in self._entries.items()
                if _canonical_value(lut.params.turbidity) == target
            ]
            for key in removed:
                self._entries.pop(key, None)
                self._order.remove(key)
            return removed

    def invalidate_all(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._order.clear()
            return removed


# ---------------------------------------------------------------------------
# Canonical hashing
# ---------------------------------------------------------------------------


def _canonical_value(value: Any) -> int:
    return int(round(float(value) / HASH_QUANTUM))


def _canonical_params(params: AtmosphereParams) -> dict[str, Any]:
    return {
        "version": HASH_VERSION,
        "altitude_m": _canonical_value(params.altitude_m),
        "turbidity": _canonical_value(params.turbidity),
        "sun_elevation_deg": _canonical_value(params.sun_elevation_deg),
        "sun_azimuth_deg": _canonical_value(params.sun_azimuth_deg),
        "lut_sun_samples": int(params.lut_sun_samples),
        "lut_view_samples": int(params.lut_view_samples),
    }


def param_hash(value: Any = None, **overrides: Any) -> str:
    """Stable precomputation key for the normalised parameters.

    The digest covers only canonical, quantised parameter values.  Dict
    insertion order is irrelevant (keys are sorted), ``hash()`` is not used
    (so ``PYTHONHASHSEED`` cannot change it) and no timestamp is included.
    """

    params = _apply_overrides(_coerce_params(value), overrides)
    payload = json.dumps(_canonical_params(params), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Default shared state
# ---------------------------------------------------------------------------


_CACHE = LUTCache()
_FENCE = WriteFence()


def _seed_published_fence() -> dict[str, Any]:
    """Ensure the shared fence has a published table and return its state."""

    if _FENCE.snapshot() is None:
        lut = _CACHE.get_or_build(AtmosphereParams())
        _FENCE.commit(lut)
    return _FENCE.read()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def altitude_recalc(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Recompute scattering coefficients for the requested altitude.

    Returns the sea-level reference coefficients, the coefficients at the new
    altitude, and a LUT built with them.  Changing the altitude therefore
    cannot leave a LUT bound to the previous scattering coefficients.
    """

    params = _apply_overrides(_coerce_params(value, "altitude_m"), overrides)
    errors = _param_errors(params)
    effective = params if not errors else _clamped_params(params)

    current = _coefficients(effective)
    reference = _coefficients(replace(effective, altitude_m=0.0))
    lut = _build_lut(effective)

    rayleigh0 = np.asarray(reference["rayleigh"], dtype=np.float64)
    rayleigh1 = np.asarray(current["rayleigh"], dtype=np.float64)
    ratio = float(np.mean(rayleigh1 / np.maximum(rayleigh0, EPS)))

    return {
        "ok": not errors,
        "errors": errors,
        "altitude_m": effective.altitude_m,
        "params": _params_dict(effective),
        "key": lut.key,
        "checksum": lut.checksum,
        "rayleigh": tuple(current["rayleigh"]),
        "mie": current["mie"],
        "mie_extinction": current["mie_extinction"],
        "rayleigh_sea_level": tuple(reference["rayleigh"]),
        "pressure_ratio": current["pressure_ratio"],
        "density_ratio": current["density_ratio"],
        "scatter_ratio_vs_sea_level": ratio,
        "recomputed": True,
        "lut": lut,
    }


def lut_smooth(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Blur the horizon rows and report the banding before and after."""

    lut = _as_lut(_apply_overrides(_coerce_params(value), overrides))
    elevations = np.asarray(lut.sun_elevations_deg, dtype=np.float64)
    banding_before = _banding_metric(lut.data, elevations)
    smoothed = _with_horizon_smoothing(lut)
    banding_after = _banding_metric(smoothed.data, elevations)

    return {
        "key": smoothed.key,
        "checksum": smoothed.checksum,
        "banding_before": banding_before,
        "banding_after": banding_after,
        "banding_reduction": banding_before - banding_after,
        "smoothed": smoothed.smoothed,
        "sigma_deg": HORIZON_SIGMA_DEG,
        "band_deg": HORIZON_BAND_DEG,
        "lut": smoothed,
    }


def write_fence(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Stage a LUT asynchronously and publish it with an ordered fence."""

    params = _apply_overrides(_coerce_params(value, "altitude_m"), overrides)
    lut = _build_lut(params)

    before = _FENCE.read()
    staged_generation = _FENCE.stage(lut)
    during = _FENCE.read()
    committed_generation = _FENCE.commit()
    after = _FENCE.read()

    return {
        "staged_generation": staged_generation,
        "committed_generation": committed_generation,
        "before": before,
        "during": during,
        "after": after,
        "published_after_commit": after["published"],
        "old_snapshot_was_visible_until_commit": during["key"] == before["key"],
        "atomic_swap": committed_generation == staged_generation + 1,
        "key": lut.key,
        "checksum": lut.checksum,
        "fence": _FENCE,
    }


def turbidity_invalidate(value: Any = None, new_turbidity: float | None = None, **overrides: Any) -> dict[str, Any]:
    """Change turbidity and evict every table computed with the old value."""

    params = _apply_overrides(_coerce_params(value, "turbidity"), overrides)
    old_turbidity = params.turbidity
    target = float(new_turbidity) if new_turbidity is not None else old_turbidity + 1.0
    target = _clamp(target, 0.0, 20.0)

    # Guarantee the cache holds a table keyed with the old turbidity.
    old_lut = _CACHE.get_or_build(params)
    other = replace(params, turbidity=_clamp(old_turbidity - 0.25, 0.0, 20.0))
    if other.turbidity != old_turbidity:
        _CACHE.get_or_build(other)

    before_keys = set(_CACHE.keys())
    removed = _CACHE.invalidate_turbidity(old_turbidity)
    new_params = replace(params, turbidity=target)
    new_lut = _CACHE.get_or_build(new_params)
    if _FENCE.snapshot() is not None and _FENCE.snapshot().key == old_lut.key:
        _FENCE.invalidate("turbidity")

    remaining = set(_CACHE.keys())
    stale_remaining = sorted(_CACHE.stale_keys(old_turbidity))

    return {
        "changed": _canonical_value(target) != _canonical_value(old_turbidity),
        "old_turbidity": old_turbidity,
        "new_turbidity": target,
        "old_key": old_lut.key,
        "new_key": new_lut.key,
        "key_changed": old_lut.key != new_lut.key,
        "removed": removed,
        "removed_count": len(removed),
        "cache_size_before": len(before_keys),
        "cache_size_after": len(remaining),
        "stale_remaining": stale_remaining,
        "stale_served": bool(stale_remaining),
        "new_checksum": new_lut.checksum,
        "lut": new_lut,
    }


def sample_determinism(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Sample the same direction twice and confirm bit-identical colours."""

    params = _apply_overrides(_coerce_params(value, "sun_elevation_deg"), overrides)
    if _param_errors(params):
        params = _clamped_params(params)
    lut = _CACHE.get_or_build(params)

    elevation = _quantize(params.sun_elevation_deg, INPUT_QUANTUM)
    zenith = 45.0
    repeats = int(overrides.get("repeats", 3) or 3)
    first = _sample_lut(lut, elevation, zenith)
    samples = [first] + [_sample_lut(lut, elevation, zenith) for _ in range(repeats - 1)]
    nudged = _sample_lut(lut, elevation + SAMPLE_QUANTUM * 0.25, zenith)

    return {
        "key": lut.key,
        "elevation": elevation,
        "view_zenith_deg": zenith,
        "samples": samples,
        "identical": all(sample == first for sample in samples),
        "nudged_within_quantum_identical": nudged == first,
        "quantum": SAMPLE_QUANTUM,
        "input_quantum": INPUT_QUANTUM,
        "deterministic": True,
    }


def sun_angle(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Normalise the sun angle and return its direction and air mass."""

    params = _apply_overrides(_coerce_params(value, "sun_elevation_deg"), overrides)
    elevation = _clamp(params.sun_elevation_deg, -90.0, 90.0)
    azimuth = params.sun_azimuth_deg % 360.0
    direction = _sun_direction(elevation, azimuth)
    return {
        "sun_elevation_deg": elevation,
        "sun_azimuth_deg": azimuth,
        "direction": direction,
        "airmass": _airmass(elevation),
        "above_horizon": elevation > 0.0,
        "at_horizon": abs(elevation) <= 0.5,
        "twilight": -18.0 <= elevation <= -0.833,
        "key": param_hash(replace(params, sun_elevation_deg=elevation, sun_azimuth_deg=azimuth)),
    }


def validation(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Check parameters against the supported ranges."""

    params = _apply_overrides(_coerce_params(value), overrides)
    errors = _param_errors(params)
    warnings: list[str] = []
    if not errors:
        if params.turbidity > 10.0:
            warnings.append("turbidity above 10 is unusually hazy")
        if abs(params.altitude_m) > 50_000.0:
            warnings.append("altitude above 50 km leaves the troposphere")
    return {
        "ok": not errors,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "params": _params_dict(params),
    }


def audit(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Run the invariants that the six original defects violated."""

    params = _apply_overrides(_coerce_params(value), overrides)
    if _param_errors(params):
        params = _clamped_params(params)
    elevations = np.asarray(_sun_elevation_nodes(params.lut_sun_samples), dtype=np.float64)

    base = _build_lut(params)
    banding_before = _banding_metric(base.data, elevations)
    smoothed = _with_horizon_smoothing(base)
    banding_after = _banding_metric(smoothed.data, elevations)

    low = _coefficients(replace(params, altitude_m=0.0))
    high = _coefficients(replace(params, altitude_m=5000.0))

    reordered = dict(reversed(list(_params_dict(params).items())))
    fence_generation = _FENCE.generation
    _FENCE.stage(smoothed)
    fence_staged = _FENCE.generation
    _FENCE.commit()
    fence_committed = _FENCE.generation

    checks = {
        "altitude_recomputes_coefficients": high["rayleigh"] != low["rayleigh"],
        "smoothing_reduces_banding": banding_after <= banding_before + 1e-12,
        "fence_is_ordered": fence_committed == fence_staged + 1 and fence_staged == fence_generation,
        "turbidity_changes_key": param_hash(replace(params, turbidity=params.turbidity + 0.5)) != param_hash(params),
        "hash_is_order_independent": param_hash(reordered) == param_hash(params),
        "lut_checksum_is_stable": _build_lut(params).checksum == base.checksum,
        "table_shape_matches_grid": base.data.shape
        == (params.lut_sun_samples, params.lut_view_samples, 3),
    }

    return {
        "ok": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "banding_before": banding_before,
        "banding_after": banding_after,
        "fence_generation": fence_committed,
        "key": base.key,
        "checksum": base.checksum,
    }


def _clamped_params(raw: AtmosphereParams) -> AtmosphereParams:
    """Clamp a possibly invalid parameter set into the supported ranges."""

    return AtmosphereParams(
        altitude_m=_clamp(raw.altitude_m if math.isfinite(raw.altitude_m) else 0.0, 0.0, 100_000.0),
        turbidity=_clamp(raw.turbidity if math.isfinite(raw.turbidity) else 2.0, 0.0, 20.0),
        sun_elevation_deg=_clamp(
            raw.sun_elevation_deg if math.isfinite(raw.sun_elevation_deg) else 10.0, -90.0, 90.0
        ),
        sun_azimuth_deg=(
            float(raw.sun_azimuth_deg) % 360.0 if math.isfinite(raw.sun_azimuth_deg) else 0.0
        ),
        lut_sun_samples=int(_clamp(raw.lut_sun_samples, MIN_SAMPLES, MAX_SAMPLES)),
        lut_view_samples=int(_clamp(raw.lut_view_samples, MIN_SAMPLES, MAX_SAMPLES)),
    )


def fallback(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Clamp invalid parameters to the nearest supported state."""

    raw = _apply_overrides(_coerce_params(value, "altitude_m"), overrides)
    errors = _param_errors(raw)
    clamped = _clamped_params(raw)
    lut = _build_lut(clamped)
    return {
        "used_fallback": bool(errors),
        "errors": errors,
        "params": _params_dict(clamped),
        "params_typed": clamped,
        "key": lut.key,
        "checksum": lut.checksum,
        "lut": lut,
    }


def precompute(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Build (or reuse) the table for the parameters and cache it."""

    params = _apply_overrides(_coerce_params(value, "altitude_m"), overrides)
    if _param_errors(params):
        params = _clamped_params(params)
    cached_before = _CACHE.get(params) is not None
    lut = _CACHE.get_or_build(params)
    return {
        "key": lut.key,
        "checksum": lut.checksum,
        "params": _params_dict(lut.params),
        "sun_samples": int(lut.sun_elevations_deg.size),
        "view_samples": int(lut.view_zeniths_deg.size),
        "cache_hit": cached_before,
        "cache_size": len(_CACHE),
        "altitude_m": lut.params.altitude_m,
        "turbidity": lut.params.turbidity,
        "rayleigh": _coefficients(lut.params)["rayleigh"],
        "singleton_cached": _CACHE.get(params) is lut,
        "lut": lut,
    }


def metrics(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Physically meaningful summary numbers for the current table."""

    params = _apply_overrides(_coerce_params(value), overrides)
    if _param_errors(params):
        params = _clamped_params(params)
    lut = _CACHE.get_or_build(params)
    elevations = np.asarray(lut.sun_elevations_deg, dtype=np.float64)
    data = np.asarray(lut.data, dtype=np.float64)

    transmittance = _sun_transmittance(params, params.sun_elevation_deg)
    smoothed = _with_horizon_smoothing(lut)
    return {
        "key": lut.key,
        "checksum": lut.checksum,
        "banding": _banding_metric(data, elevations),
        "banding_smoothed": _banding_metric(smoothed.data, elevations),
        "horizon_gradient": _gradient_metric(data, elevations),
        "mean_radiance": float(np.mean(data)),
        "max_radiance": float(np.max(data)),
        "min_radiance": float(np.min(data)),
        "dynamic_range": float(np.max(data) - np.min(data)),
        "sun_transmittance": transmittance,
        "airmass": _airmass(params.sun_elevation_deg),
        "rayleigh": _coefficients(params)["rayleigh"],
        "mie": _coefficients(params)["mie"],
        "pressure_ratio": _coefficients(params)["pressure_ratio"],
        "non_negative": bool(np.all(data >= 0.0)),
        "finite": bool(np.all(np.isfinite(data))),
    }


def weather_switch(value: Any = None, turbidity: float | None = None, **overrides: Any) -> dict[str, Any]:
    """Switch weather and prove the old table is no longer served."""

    params = _apply_overrides(_coerce_params(value, "turbidity"), overrides)
    old_turbidity = params.turbidity
    target = None
    for candidate in (turbidity, overrides.get("new_turbidity"), overrides.get("to_turbidity")):
        if candidate is not None:
            target = float(candidate)
            break
    if target is None:
        target = old_turbidity + 1.0
    target = _clamp(target, 0.0, 20.0)

    old_lut = _CACHE.get_or_build(params)
    removed = _CACHE.invalidate_turbidity(old_turbidity)
    if _FENCE.snapshot() is not None and _FENCE.snapshot().key == old_lut.key:
        _FENCE.invalidate("weather")

    new_params = replace(params, turbidity=target)
    new_lut = _CACHE.get_or_build(new_params)
    old_after = _CACHE.get(old_lut.params)

    return {
        "changed": _canonical_value(target) != _canonical_value(old_turbidity),
        "old_turbidity": old_turbidity,
        "new_turbidity": target,
        "old_key": old_lut.key,
        "new_key": new_lut.key,
        "new_checksum": new_lut.checksum,
        "invalidated": removed,
        "old_key_still_cached": old_after is not None,
        "stale_served": old_after is not None,
        "cache_size": len(_CACHE),
        "new_lut": new_lut,
    }


def skybox_bind(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Bind the published snapshot; readers never see a staged table."""

    state = _seed_published_fence()
    if value is not None or overrides:
        params = _apply_overrides(_coerce_params(value, "altitude_m"), overrides)
        lut = _CACHE.get_or_build(params)
        _FENCE.stage(lut)
        published_before_commit = _FENCE.read()
        generation = _FENCE.commit()
    else:
        published_before_commit = state
        generation = _FENCE.generation

    snapshot = _FENCE.snapshot()
    assert snapshot is not None
    publish = lut if (value is not None or overrides) else snapshot
    return {
        "generation": generation,
        "key": publish.key,
        "checksum": publish.checksum,
        "published": True,
        "stable": _FENCE.generation == generation,
        "published_before_commit_generation": published_before_commit["generation"],
        "commit_is_newer": generation == published_before_commit["generation"] + 1
        if (value is not None or overrides)
        else True,
        "width": int(publish.sun_elevations_deg.size),
        "height": int(publish.view_zeniths_deg.size),
        "sun_elevations_deg": tuple(float(x) for x in publish.sun_elevations_deg[:3]),
        "fence": _FENCE,
    }


def determinism(value: Any = None, **overrides: Any) -> dict[str, Any]:
    """Build and sample twice from scratch; results must be bit-identical."""

    params = _apply_overrides(_coerce_params(value), overrides)
    if _param_errors(params):
        params = _clamped_params(params)

    first = _build_lut(params)
    second = _build_lut(replace(params))
    angle = _quantize(params.sun_elevation_deg, INPUT_QUANTUM)
    sample_first = _sample_lut(first, angle, 45.0)
    sample_second = _sample_lut(second, angle, 45.0)

    return {
        "identical": first.checksum == second.checksum and sample_first == sample_second,
        "checksum_first": first.checksum,
        "checksum_second": second.checksum,
        "hash_first": first.key,
        "hash_second": second.key,
        "sample_first": sample_first,
        "sample_second": sample_second,
        "samples_identical": sample_first == sample_second,
        "key": first.key,
        "deterministic": first.checksum == second.checksum and sample_first == sample_second,
    }


def sample(value: Any, sun_elevation_deg: float, view_zenith_deg: float, **overrides: Any) -> tuple[float, float, float]:
    """Convenience deterministic sample entry point."""

    if isinstance(value, AtmosphereLUT):
        lut = value
    else:
        lut = _as_lut(_apply_overrides(_coerce_params(value), overrides))
    if not lut.smoothed:
        lut = _with_horizon_smoothing(lut)
    return _sample_lut(
        lut,
        _quantize(sun_elevation_deg, INPUT_QUANTUM),
        _quantize(view_zenith_deg, INPUT_QUANTUM),
    )
