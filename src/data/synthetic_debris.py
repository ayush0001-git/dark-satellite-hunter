"""
Synthetic Debris Injection Module (`synthetic_debris.py`)

Physics-based model of the optical signature of a tumbling, non-cooperative
orbital object (rocket body, defunct satellite, fragmentation debris):

1.  A diffuse (quasi-Lambertian) component: as the body tumbles with rotation
    period P, the projected illuminated area seen by the observer varies,
    producing a periodic magnitude modulation of peak-to-peak `amplitude` mag.
2.  Sharp specular glints: when a flat, reflective facet (solar panel, MLI
    blanket, antenna dish) momentarily aligns with the Sun-observer bisector,
    the object brightens by several magnitudes for a single exposure.
    Glint brightening amplitudes are drawn from an exponential distribution,
    consistent with observed GEO glint statistics (Murakami+ 2021).

All arithmetic is carried out in flux space and converted back to the
astronomical magnitude system (mag = -2.5 * log10(flux)), so the injected
light curves have physically sensible magnitudes.
"""
import numpy as np
from typing import Optional, Tuple, Union

RngLike = Union[np.random.Generator, int, None]


def _as_rng(seed_or_rng: RngLike) -> np.random.Generator:
    """Accepts a Generator, an int seed, or None and returns a Generator."""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def tumbling_glint_model(
    t: np.ndarray,
    period: float = 0.25,
    amplitude: float = 1.5,
    glint_prob: float = 0.05,
    glint_scale_mag: float = 1.5,
    max_glint_mag: float = 4.0,
    phase0: Optional[float] = None,
    seed: RngLike = None,
    band: Optional[np.ndarray] = None,
    material_type: str = "Kapton_MLI",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnitude offset of a tumbling object relative to its mean magnitude,
    incorporating multi-band Material BRDF Chromaticity (`g` vs `r` band color shifts).

    Args:
        t: Observation epochs (days; MJD or days-since-t0 both fine).
        period: Tumbling (rotation) period in days.
        amplitude: Peak-to-peak diffuse modulation in magnitudes.
        glint_prob: Per-epoch probability that a specular glint is caught.
        glint_scale_mag: Mean of the exponential glint-brightening distribution (mag).
        max_glint_mag: Physical cap on glint brightening (mag).
        phase0: Rotational phase at t=0 in [0, 1); random if None.
        seed: int seed or np.random.Generator for reproducibility.
        band: Optional array of filter indices (`0=g`, `1=r`).
        material_type: Surface material governing specular chromatic shift (`Kapton_MLI`, `Solar_Cell`, `Aluminum_Bus`).

    Returns:
        delta_mag: Magnitude offset per epoch. Negative = brighter.
        glint_mask: Boolean array, True where a specular glint occurred.
    """
    rng = _as_rng(seed)
    t = np.asarray(t, dtype=np.float64)
    period = max(float(period), 1e-6)

    if phase0 is None:
        phase0 = float(rng.uniform(0.0, 1.0))

    # Diffuse tumbling term: fundamental + weak second harmonic (zero-mean by design)
    phase = (t / period + phase0) % 1.0
    diffuse = 0.5 * amplitude * (
        np.sin(2.0 * np.pi * phase) + 0.3 * np.sin(4.0 * np.pi * phase)
    )

    # Specular glints: brief brightening events (negative delta magnitude)
    glint_mask = rng.random(t.shape) < glint_prob
    glint_amp = np.minimum(
        rng.exponential(scale=glint_scale_mag, size=t.shape), max_glint_mag
    )

    # Apply Material BRDF Chromatic Glint Shift across optical bands (if provided)
    if band is not None and len(band) == len(t):
        band_arr = np.asarray(band, dtype=np.int64)
        # Kapton/MLI thermal blankets reflect red/NIR stronger than blue (r band brighter spike)
        if material_type == "Kapton_MLI":
            chromatic_factor = np.where(band_arr == 1, 1.35, 0.90)
        # Silicon Solar Cells reflect blue/UV strongly at high incidence (g band brighter spike)
        elif material_type == "Solar_Cell":
            chromatic_factor = np.where(band_arr == 0, 1.25, 0.95)
        else:
            chromatic_factor = np.ones_like(t, dtype=np.float64)
        glint_amp = glint_amp * chromatic_factor

    delta_mag = diffuse - glint_amp * glint_mask

    return delta_mag.astype(np.float64), glint_mask


def inject_debris_into_lightcurve(
    t: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    period: float = 0.25,
    amplitude: float = 1.5,
    glint_prob: float = 0.08,
    debris_mean_mag: Optional[float] = None,
    seed: RngLike = None,
    band: Optional[np.ndarray] = None,
    material_type: str = "Kapton_MLI",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Blends the flux of a synthetic tumbling object into a background light curve,
    preserving multi-band chromatic glint ratios and fast vectorized arithmetic.

    Args:
        t: Observation epochs (days).
        mag: Background-source magnitudes.
        magerr: Background-source magnitude uncertainties.
        period: Tumbling period in days.
        amplitude: Peak-to-peak diffuse modulation (mag).
        glint_prob: Per-epoch specular glint probability.
        debris_mean_mag: Mean magnitude of the debris.
        seed: int seed or np.random.Generator.
        band: Optional filter array (`0=g`, `1=r`).
        material_type: Surface material (`Kapton_MLI`, `Solar_Cell`, `Aluminum_Bus`).

    Returns:
        (t, new_mag, new_magerr, glint_mask)
    """
    rng = _as_rng(seed)
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    magerr = np.asarray(magerr, dtype=np.float64)

    if debris_mean_mag is None:
        debris_mean_mag = float(np.median(mag))

    delta_mag, glint_mask = tumbling_glint_model(
        t, period=period, amplitude=amplitude, glint_prob=glint_prob, seed=rng, band=band, material_type=material_type
    )

    # Vectorized fast flux-space combination relative to mag=0 zero point
    base_flux = np.power(10.0, -0.4 * mag)
    debris_flux = np.power(10.0, -0.4 * (debris_mean_mag + delta_mag))
    combined_flux = base_flux + debris_flux

    new_mag = -2.5 * np.log10(combined_flux)

    # First-order error propagation
    base_flux_err = base_flux * (np.log(10.0) / 2.5) * magerr
    new_magerr = (2.5 / np.log(10.0)) * (base_flux_err / combined_flux)
    new_magerr = np.clip(new_magerr, 0.005, 0.5)

    return t, new_mag.astype(np.float64), new_magerr.astype(np.float64), glint_mask


def compute_astrometric_centroid_shift(
    mag_star: np.ndarray,
    mag_debris: np.ndarray,
    separation_arcsec: float = 1.2,
) -> np.ndarray:
    """
    Computes the astrometric centroid pull (`center of light` displacement in arcsec)
    caused by blending a faint background star (`mag_star`) with a nearby moving
    or glinting debris interloper (`mag_debris`) within the camera PSF (`~1.5 arcsec`).

    Args:
        mag_star: Background star magnitude per epoch.
        mag_debris: Debris interloper magnitude per epoch.
        separation_arcsec: Angular separation between true star and true debris (< PSF FWHM).

    Returns:
        centroid_shift_arcsec: Astrometric centroid displacement toward the debris.
    """
    mag_star = np.asarray(mag_star, dtype=np.float64)
    mag_debris = np.asarray(mag_debris, dtype=np.float64)

    flux_star = np.power(10.0, -0.4 * mag_star)
    flux_debris = np.power(10.0, -0.4 * mag_debris)
    flux_total = flux_star + flux_debris

    # Flux-weighted centroid offset vector magnitude
    shift_arcsec = (flux_debris / np.maximum(flux_total, 1e-12)) * float(separation_arcsec)
    return shift_arcsec.astype(np.float64)
