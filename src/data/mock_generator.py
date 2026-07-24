"""
Mock Survey Generator (`mock_generator.py`)

Simulates a small ZTF-like optical survey field with known ground truth:

* Clean astrophysical variables and non-variables (RR Lyrae, eclipsing
  binaries, damped-random-walk QSOs, constant stars) at fixed sky positions.
* Uncataloged tumbling debris ("dark satellites") whose flux is blended into
  faint background sources and whose positions drift slowly across the field,
  mimicking quasi-geostationary graveyard-belt objects.

Design notes for scientific integrity
-------------------------------------
The generator also produces a *reference catalog* (`build_reference_catalog`)
containing only the cataloged astrophysical sources - the mock analogue of
Gaia/Simbad. Downstream discovery code must identify debris purely from the
photometric/astrometric data plus positional cross-matching against this
catalog; ground-truth `label` fields exist ONLY for validation and are never
consulted by the discovery pipeline (see tests/test_no_label_leakage.py).

All randomness flows through a single `numpy.random.Generator`, so a given
seed reproduces the survey exactly.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

from .synthetic_debris import compute_astrometric_centroid_shift, inject_debris_into_lightcurve

# Mock field geometry (deg). A single ZTF readout channel is ~0.85 deg on a
# side; we use a slightly larger box for headroom.
FIELD_RA_DEG = 120.0
FIELD_DEC_DEG = 35.0
FIELD_HALF_WIDTH_DEG = 1.5

# Per-epoch astrometric scatter for cataloged point sources (arcsec).
ASTROMETRIC_JITTER_ARCSEC = 0.3


def generate_irregular_mjd(
    rng: np.random.Generator,
    n_points: int = 150,
    start_mjd: float = 59000.0,
    duration_days: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ZTF-like irregular observation epochs: night-time only, weather gaps,
    plus random g/r band assignment (0 = g, 1 = r).
    """
    # Oversample, then keep night-time epochs (UTC fractional day near 0/1).
    raw_times = np.sort(rng.uniform(start_mjd, start_mjd + duration_days, size=n_points * 3))
    frac_day = raw_times % 1.0
    night_mask = (frac_day < 0.17) | (frac_day > 0.83)
    valid_times = raw_times[night_mask]

    if len(valid_times) >= n_points:
        keep = rng.choice(len(valid_times), size=n_points, replace=False)
        valid_times = np.sort(valid_times[keep])
    # else: sparse night coverage; keep everything we have.

    # g-band ~45%, r-band ~55% (ZTF public-survey mix).
    bands = (rng.random(len(valid_times)) > 0.45).astype(np.int64)
    return valid_times, bands


def _robust_scale_per_band(
    mag: np.ndarray, magerr: np.ndarray, band: np.ndarray, eps: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """Median/IQR normalization applied independently per photometric band."""
    norm_mag = np.zeros_like(mag, dtype=np.float64)
    norm_err = np.zeros_like(magerr, dtype=np.float64)
    for b in np.unique(band):
        sel = band == b
        med = np.median(mag[sel])
        q75, q25 = np.percentile(mag[sel], [75, 25])
        iqr = (q75 - q25) + eps
        norm_mag[sel] = (mag[sel] - med) / iqr
        norm_err[sel] = magerr[sel] / iqr
    return norm_mag, norm_err


def _star_ephemeris(
    rng: np.random.Generator, n_epochs: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Fixed sky position plus per-epoch astrometric measurement scatter."""
    ra_true = FIELD_RA_DEG + rng.uniform(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG)
    dec_true = FIELD_DEC_DEG + rng.uniform(-FIELD_HALF_WIDTH_DEG, FIELD_HALF_WIDTH_DEG)
    jitter_deg = ASTROMETRIC_JITTER_ARCSEC / 3600.0
    cosd = max(np.cos(np.radians(dec_true)), 1e-6)
    ra = ra_true + rng.normal(0.0, jitter_deg / cosd, size=n_epochs)
    dec = dec_true + rng.normal(0.0, jitter_deg, size=n_epochs)
    return ra, dec, ra_true, dec_true


def _debris_ephemeris(
    rng: np.random.Generator, t_days: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Slowly drifting track across the field: linear motion at a rate typical
    of graveyard-belt drifters as seen night-to-night in a sidereal frame.

    NOTE (simplification): real GEO debris sweeps a full 360 deg of RA per
    sidereal day; a fixed-field survey would see it as one streak per night.
    We model only the slow envelope drift so that repeated photometry of the
    same object is possible - this is the standing simplification of the
    mock survey and is documented in the README's Limitations section.
    """
    rate_deg_day = rng.uniform(0.005, 0.012) * rng.choice([-1.0, 1.0])
    theta = rng.uniform(0.0, 2.0 * np.pi)
    span = float(t_days[-1] - t_days[0]) if len(t_days) > 1 else 1.0
    max_drift = abs(rate_deg_day) * span
    # Start opposite the drift direction so the track stays inside the field.
    ra0 = FIELD_RA_DEG - 0.5 * max_drift * np.cos(theta) + rng.uniform(-0.8, 0.8)
    dec0 = FIELD_DEC_DEG - 0.5 * max_drift * np.sin(theta) + rng.uniform(-0.8, 0.8)

    jitter_deg = ASTROMETRIC_JITTER_ARCSEC / 3600.0
    ra = ra0 + rate_deg_day * np.cos(theta) * t_days + rng.normal(0, jitter_deg, len(t_days))
    dec = dec0 + rate_deg_day * np.sin(theta) * t_days + rng.normal(0, jitter_deg, len(t_days))
    return ra, dec, abs(rate_deg_day)


def generate_mock_ztf_dataset(
    n_clean: int = 400,
    n_debris: int = 100,
    seq_len: int = 256,
    seed: int = 42,
    shuffle: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Builds the mock survey.

    Args:
        n_clean: Number of cataloged astrophysical sources.
        n_debris: Number of uncataloged tumbling-debris blends.
        seq_len: Kept for API compatibility (windowing happens in the Dataset).
        seed: Master seed; the same seed reproduces the survey exactly.
        shuffle: Randomize record order (recommended - avoids accidental
                 class-ordered splits downstream).

    Returns:
        List of per-object records with keys:
            object_id, t (days since first epoch), mag / magerr (per-band
            robust-normalized), band (0=g, 1=r), raw_mag, raw_magerr,
            raw_t (MJD), ra, dec (deg, per epoch),
            label (0 clean / 1 debris - VALIDATION ONLY),
            true_class, period_guess (days; simulation truth),
            drift_rate_deg_day (debris only).
    """
    rng = np.random.default_rng(seed)
    dataset: List[Dict[str, np.ndarray]] = []

    # --- 1. Cataloged astrophysical sources -------------------------------
    for i in range(n_clean):
        n_pts = int(rng.integers(80, 220))
        mjd, band = generate_irregular_mjd(rng, n_points=n_pts)
        t_days = mjd - mjd[0]

        astro_class = rng.choice(
            ["RR_Lyrae", "Eclipsing_Binary", "QSO", "Constant_Star"],
            p=[0.25, 0.20, 0.25, 0.30],
        )
        base_mag = rng.uniform(15.5, 19.5)
        magerr = rng.uniform(0.01, 0.05, size=len(mjd))
        period = 0.0

        if astro_class == "RR_Lyrae":
            period = rng.uniform(0.4, 0.8)
            phase = (mjd / period) % 1.0
            mag_curve = base_mag - 0.6 * np.sin(2 * np.pi * phase) - 0.2 * np.sin(4 * np.pi * phase)
        elif astro_class == "Eclipsing_Binary":
            period = rng.uniform(1.2, 5.0)
            phase = (mjd / period) % 1.0
            mag_curve = np.full(len(mjd), base_mag)
            dip = (phase > 0.45) & (phase < 0.55)
            mag_curve[dip] += rng.uniform(0.5, 1.2)
        elif astro_class == "QSO":
            mag_curve = base_mag + np.cumsum(rng.normal(0, 0.03, size=len(mjd)))
        else:  # Constant_Star
            mag_curve = base_mag + np.zeros(len(mjd))

        # Static per-source color term and photometric noise.
        color_offset = rng.uniform(-0.4, 0.8)
        mag_curve = mag_curve - color_offset * (band == 1)
        mag_curve = mag_curve + rng.normal(0.0, magerr)

        ra, dec, ra_true, dec_true = _star_ephemeris(rng, len(mjd))
        norm_mag, norm_err = _robust_scale_per_band(mag_curve, magerr, band)

        dataset.append({
            "object_id": f"ZTF_MOCK_{i:05d}",
            "t": t_days.astype(np.float32),
            "mag": norm_mag.astype(np.float32),
            "magerr": norm_err.astype(np.float32),
            "band": band,
            "raw_mag": mag_curve.astype(np.float32),
            "raw_magerr": magerr.astype(np.float32),
            "raw_t": mjd.astype(np.float32),
            "ra": ra.astype(np.float64),
            "dec": dec.astype(np.float64),
            "ra_true": float(ra_true),
            "dec_true": float(dec_true),
            "label": np.int64(0),
            "true_class": str(astro_class),
            "period_guess": float(period),
        })

    # --- 2. Uncataloged tumbling debris blends ----------------------------
    for i in range(n_debris):
        n_pts = int(rng.integers(100, 240))
        mjd, band = generate_irregular_mjd(rng, n_points=n_pts)
        t_days = mjd - mjd[0]

        # Faint background source near the detection limit...
        bg_mag = rng.uniform(19.0, 20.5)
        bg_curve = bg_mag + rng.normal(0, 0.03, size=len(mjd))
        magerr = rng.uniform(0.02, 0.06, size=len(mjd))

        # ...with a brighter tumbling interloper blended in.
        tumble_period = rng.uniform(0.05, 0.5)          # 1.2 - 12 h
        glint_p = rng.uniform(0.05, 0.15)
        debris_mag = rng.uniform(16.5, 18.5)
        material_choice = str(rng.choice(["Kapton_MLI", "Solar_Cell", "Aluminum_Bus"]))
        _, new_mag, new_err, glints = inject_debris_into_lightcurve(
            mjd, bg_curve, magerr,
            period=tumble_period, amplitude=1.8, glint_prob=glint_p,
            debris_mean_mag=debris_mag, seed=rng, band=band, material_type=material_choice
        )

        ra, dec, drift_rate = _debris_ephemeris(rng, t_days)

        # Astrometric centroid wobble: the measured position of the blend is
        # the flux-weighted centroid of the static background star and the
        # debris interloper inside the PSF. When the debris glints, the
        # centroid snaps toward it - a real SDA signature the discovery
        # features can exploit (astrometry-photometry correlation).
        debris_flux = np.maximum(
            np.power(10.0, -0.4 * new_mag) - np.power(10.0, -0.4 * bg_curve), 1e-12
        )
        debris_only_mag = -2.5 * np.log10(debris_flux)
        blend_sep_arcsec = float(rng.uniform(0.4, 1.4))  # star-debris offset inside PSF
        centroid_shift = compute_astrometric_centroid_shift(
            bg_curve, debris_only_mag, separation_arcsec=blend_sep_arcsec
        )
        # The shift is measured FROM the star TOWARD the debris; equivalently
        # the centroid sits (blend_sep - shift) away from the debris track.
        wobble_deg = (blend_sep_arcsec - centroid_shift) / 3600.0
        wobble_theta = rng.uniform(0.0, 2.0 * np.pi)
        cosd = max(np.cos(np.radians(FIELD_DEC_DEG)), 1e-6)
        ra = ra - wobble_deg * np.cos(wobble_theta) / cosd
        dec = dec - wobble_deg * np.sin(wobble_theta)

        norm_mag, norm_err = _robust_scale_per_band(new_mag, new_err, band)

        dataset.append({
            "object_id": f"ZTF_MOCK_D{i:04d}",
            "t": t_days.astype(np.float32),
            "mag": norm_mag.astype(np.float32),
            "magerr": norm_err.astype(np.float32),
            "band": band,
            "raw_mag": new_mag.astype(np.float32),
            "raw_magerr": new_err.astype(np.float32),
            "raw_t": mjd.astype(np.float32),
            "ra": ra.astype(np.float64),
            "dec": dec.astype(np.float64),
            "ra_true": float(np.mean(ra)),
            "dec_true": float(np.mean(dec)),
            "label": np.int64(1),
            "true_class": "DEBRIS_GLINT",
            "true_material": material_choice,
            "period_guess": float(tumble_period),
            "drift_rate_deg_day": float(drift_rate),
            "glint_mask": glints,
        })

    if shuffle:
        order = rng.permutation(len(dataset))
        dataset = [dataset[j] for j in order]

    return dataset


def build_reference_catalog(records: List[Dict]) -> List[Dict]:
    """
    Simulation-side analogue of Gaia/Simbad: the catalog of *known* sources.

    Debris objects are, by construction, absent - uncataloged objects are
    exactly what the discovery pipeline is trying to find. This function uses
    ground truth because it plays the role of the external catalog provider,
    NOT the discovery pipeline. The pipeline may only use the returned
    positions and class strings.
    """
    catalog = []
    for k, rec in enumerate(records):
        if int(rec.get("label", 0)) != 0:
            continue
        catalog.append({
            "catalog_id": f"GAIA_MOCK_{k:06d}",
            "ra_deg": float(rec["ra_true"]),
            "dec_deg": float(rec["dec_true"]),
            "obj_class": str(rec.get("true_class", "Star")),
        })
    return catalog
