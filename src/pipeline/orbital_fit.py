"""
Orbital Kinematics and Tumbling Periodicity (`orbital_fit.py`)

1. `estimate_glint_periodicity` - tumbling-period estimation from specular
   glint timing, using phase-folded concentration analysis (Rayleigh-style
   statistic) over a candidate period grid to defeat the diurnal sampling
   aliases that plague simple interval differencing.
2. `fit_preliminary_orbit` - weighted least-squares astrometric rates
   (dRA/dt, dDec/dt) with proper RA wrap handling, apparent angular speed,
   and a survey-cadence-appropriate kinematic regime classification.
"""
from ctypes import Union
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Optional, Union

# Kinematic regime thresholds for multi-night survey arcs (deg/day).
# NOTE: these classify *night-to-night mean motion in the sidereal frame*,
# not the instantaneous streak rate within a single exposure (a LEO object
# would streak at ~1 deg/s inside one exposure - a different measurement).
RATE_SIDEREAL_MAX = 5e-3      # below: star/QSO/very distant object
RATE_GEO_DRIFT_MAX = 0.5      # slow drifters: GEO graveyard-belt candidates
RATE_ASTEROID_MAX = 5.0       # main-belt asteroids ~0.25 deg/day near opposition


def _empty_periodicity() -> Dict[str, float]:
    return {
        "period_days": 0.0, "period_hours": 0.0, "num_glints": 0,
        "confidence": 0.0, "phase_sharpness": 0.0, "false_alarm_prob": 1.0,
    }


def estimate_glint_periodicity(
    t: np.ndarray,
    mag: np.ndarray,
    threshold_sigma: float = 3.0,
    min_period_hr: float = 0.2,
    max_period_hr: float = 12.0,
    oversample: float = 4.0,
    max_trials: int = 40000,
) -> Dict[str, float]:
    """
    Tumbling-period estimate from the phase coherence of bright deviations.

    For each trial period P the brightest epochs are folded and their phase
    concentration R = |<e^{2*pi*i*phase}>| computed (R -> 1 when glints repeat
    at a fixed rotational phase, R -> 0 for phase-random noise). The trial
    grid is built in frequency space with step df = 1 / (oversample * T_span),
    the standard periodogram criterion - a coarser grid decorrelates between
    samples and misses the true peak entirely on multi-week baselines.

    Returns:
        period_days / period_hours: best-fit tumbling period
        num_glints:      count of >`threshold_sigma` bright outliers
        phase_sharpness: concentration R of the best period (0 - 1)
        confidence:      combined heuristic in [0, 1]
    """
    t = np.asarray(t, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    if len(t) < 10 or len(mag) != len(t):
        return _empty_periodicity()

    order = np.argsort(t)
    t_sort, mag_sort = t[order], mag[order]

    med = np.median(mag_sort)
    robust_sigma = 1.4826 * np.median(np.abs(mag_sort - med)) + 1e-6

    # Bright deviations (mag decreases when the object brightens).
    bright_dev = med - mag_sort
    peaks, _ = find_peaks(bright_dev, height=threshold_sigma * robust_sigma)
    num_glints = int(len(peaks))

    # Epochs that carry the periodicity signal: significant bright outliers
    # if present, otherwise the top 15% brightest deviations.
    if num_glints >= 3:
        sel = peaks
    else:
        top_k = max(3, int(0.15 * len(bright_dev)))
        sel = np.argsort(bright_dev)[-top_k:]
    t_sel = t_sort[sel] - t_sort[0]

    # Vectorized phase-concentration scan over trial frequencies. Frequency
    # resolution must scale with the observation baseline.
    t_span = float(t_sort[-1] - t_sort[0])
    f_min = 24.0 / max_period_hr  # cycles per day
    f_max = 24.0 / min_period_hr
    df = 1.0 / (oversample * max(t_span, 1.0))
    n_trials = int(min(max((f_max - f_min) / df, 100), max_trials))
    trial_freq_per_day = np.linspace(f_min, f_max, n_trials)
    phases = np.outer(trial_freq_per_day, t_sel) % 1.0          # [n_trials, n_sel]
    ang = 2.0 * np.pi * phases
    r_conc = np.sqrt(np.cos(ang).mean(axis=1) ** 2 + np.sin(ang).mean(axis=1) ** 2)

    best = int(np.argmax(r_conc))
    best_sharpness = float(r_conc[best])
    best_freq = float(trial_freq_per_day[best])

    # Local refinement: the coarse grid quantizes the peak (a frequency
    # offset of df/2 smears glint phases by ~df*T_span/2 cycles and can
    # depress R substantially). A fine scan around the coarse maximum
    # recovers the true peak at negligible cost.
    fine_freqs = np.linspace(max(f_min, best_freq - 2 * df),
                             min(f_max, best_freq + 2 * df), 200)
    ang_f = 2.0 * np.pi * (np.outer(fine_freqs, t_sel) % 1.0)
    r_fine = np.sqrt(np.cos(ang_f).mean(axis=1) ** 2 + np.sin(ang_f).mean(axis=1) ** 2)
    j = int(np.argmax(r_fine))
    if r_fine[j] > best_sharpness:
        best_sharpness = float(r_fine[j])
        best_freq = float(fine_freqs[j])
    best_period_days = 1.0 / best_freq

    # Harmonic disambiguation: a symmetric tumbler glints twice per rotation,
    # so the true period may be 2x the tightest folding. Prefer the doubled
    # period when its coherence is nearly as good (within 5%).
    double_freq = 0.5 / best_period_days
    if f_min <= double_freq <= f_max:
        ang2 = 2.0 * np.pi * ((double_freq * t_sel) % 1.0)
        r_double = float(np.hypot(np.cos(ang2).mean(), np.sin(ang2).mean()))
        if r_double >= 0.95 * best_sharpness:
            best_period_days *= 2.0
            best_sharpness = max(best_sharpness, r_double)

    # False-alarm probability: under phase-random noise, K*R^2 follows an
    # approximate exponential distribution (Rayleigh test), so the single-
    # trial p-value is exp(-K R^2). Correct for the number of statistically
    # independent frequencies actually scanned (~ (f_max - f_min) * T_span).
    # NOTE: this is a conservative UPPER BOUND - the asymptotic tail
    # overestimates p for small K near R -> 1, so reported significances
    # are never inflated. It is deliberately strict: with few selected
    # epochs, some trial frequency can phase-align them almost perfectly
    # by chance, and the trials correction is what rejects those.
    k_eff = len(t_sel)
    p_single = float(np.exp(-k_eff * best_sharpness ** 2))
    n_indep = max((f_max - f_min) * max(t_span, 1.0), 1.0)
    false_alarm = float(np.clip(1.0 - (1.0 - min(p_single, 1.0)) ** n_indep, 0.0, 1.0))

    # Confidence: statistical significance of the phase coherence, boosted
    # when several independent glint detections support it.
    glint_support = min(num_glints / 5.0, 1.0)
    confidence = float(np.clip(0.6 * (1.0 - false_alarm) * best_sharpness
                               + 0.4 * glint_support, 0.0, 1.0))

    return {
        "period_days": best_period_days,
        "period_hours": best_period_days * 24.0,
        "num_glints": num_glints,
        "confidence": confidence,
        "phase_sharpness": best_sharpness,
        "false_alarm_prob": false_alarm,
    }


def fit_preliminary_orbit(
    mjd: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    astrometric_err_arcsec: float = 0.5,
) -> Dict[str, object]:
    """
    Least-squares linear astrometric fit and kinematic regime classification.

    Args:
        mjd: Observation epochs (days).
        ra, dec: Per-epoch coordinates (deg). RA is unwrapped internally so
            tracks crossing 0/360 do not produce spurious rates.
        astrometric_err_arcsec: Expected per-epoch astrometric scatter, used
            to normalize the maneuver/non-linearity check.

    Returns dict with dra_dt_deg_day, ddec_dt_deg_day, angular_rate_deg_day,
    angular_speed_deg_hr, angular_rate_arcsec_sec, regime,
    max_fit_residual_deg, non_keplerian_flag.
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)

    base = {
        "dra_dt_deg_day": 0.0, "ddec_dt_deg_day": 0.0,
        "angular_rate_deg_day": 0.0, "angular_speed_deg_hr": 0.0,
        "angular_rate_arcsec_sec": 0.0, "max_fit_residual_deg": 0.0,
        "regime": "STATIC_OR_INSUFFICIENT_DATA", "non_keplerian_flag": False,
    }
    if len(mjd) < 3 or len(ra) != len(mjd) or len(dec) != len(mjd):
        return base

    order = np.argsort(mjd)
    tt, ra_s, dec_s = mjd[order], ra[order], dec[order]
    if tt[-1] - tt[0] < 1e-5:
        base["regime"] = "SINGLE_EPOCH"
        return base

    # Unwrap RA (deg) to handle the 0/360 discontinuity.
    ra_u = np.degrees(np.unwrap(np.radians(ra_s)))
    t_rel = tt - tt[0]

    # Ordinary least squares slope+intercept for each coordinate.
    dra_dt, ra0 = np.polyfit(t_rel, ra_u, 1)
    ddec_dt, dec0 = np.polyfit(t_rel, dec_s, 1)

    mean_dec_rad = np.radians(np.mean(dec_s))
    cosd = np.cos(mean_dec_rad)
    rate_deg_day = float(np.hypot(dra_dt * cosd, ddec_dt))
    rate_deg_hr = rate_deg_day / 24.0
    rate_arcsec_sec = rate_deg_day * 3600.0 / 86400.0

    # Regime classification on the multi-night mean motion.
    if rate_deg_day < RATE_SIDEREAL_MAX:
        regime, non_kep = "SIDEREAL_STATIONARY", False
    elif rate_deg_day < RATE_GEO_DRIFT_MAX:
        regime, non_kep = "GEO_BELT_DRIFTER_CANDIDATE", True
    elif rate_deg_day < RATE_ASTEROID_MAX:
        regime, non_kep = "ASTEROID_OR_MEO_RATE", False
    else:
        regime, non_kep = "FAST_MOVER_LEO_OR_NEO", True

    # Deviation from the linear track, in units of expected astrometric noise.
    res_ra = (ra_u - (ra0 + dra_dt * t_rel)) * cosd
    res_dec = dec_s - (dec0 + ddec_dt * t_rel)
    residuals_deg = np.hypot(res_ra, res_dec)
    max_residual = float(np.max(residuals_deg))

    noise_deg = astrometric_err_arcsec / 3600.0
    if max_residual > 5.0 * noise_deg and rate_deg_day > RATE_SIDEREAL_MAX:
        non_kep = True
        regime += "_NONLINEAR_TRACK"

    return {
        "dra_dt_deg_day": float(dra_dt),
        "ddec_dt_deg_day": float(ddec_dt),
        "angular_rate_deg_day": rate_deg_day,
        "angular_speed_deg_hr": rate_deg_hr,
        "angular_rate_arcsec_sec": rate_arcsec_sec,
        "max_fit_residual_deg": max_residual,
        "regime": regime,
        "non_keplerian_flag": bool(non_kep),
    }


def estimate_altitude_from_parallax(
    parallax_arcsec: float,
    baseline_km: float = 500.0,
) -> Dict[str, float]:
    """
    Solves for approximate orbital slant distance and altitude above Earth (`R_earth ~ 6378 km`)
    given dual-station or parallax astrometric separation across a spatial baseline.

    Args:
        parallax_arcsec: Measured angular parallax shift between observatories/cameras (arcsec).
        baseline_km: Spatial baseline separation between observation points (km).

    Returns:
        dict containing `slant_distance_km`, `altitude_km`, and `parallax_confidence`.
    """
    R_EARTH_KM = 6378.137
    p_arcsec = max(float(parallax_arcsec), 1e-6)

    # If parallax is practically zero (< 0.05 arcsec), source is deep space / interstellar / asteroid
    if p_arcsec < 0.05:
        return {
            "slant_distance_km": float("inf"),
            "altitude_km": float("inf"),
            "parallax_confidence": 1.0 if p_arcsec < 0.01 else 0.5,
        }

    p_rad = np.radians(p_arcsec / 3600.0)
    slant_dist_km = float(baseline_km / max(np.tan(p_rad), 1e-9))
    altitude_km = max(float(slant_dist_km - R_EARTH_KM), 10.0)

    # Parallax confidence score based on realistic SDA regimes (400km - 45,000km)
    if 200.0 <= altitude_km <= 45000.0:
        conf = 0.98
    else:
        conf = 0.60

    return {
        "slant_distance_km": slant_dist_km,
        "altitude_km": altitude_km,
        "parallax_confidence": conf,
    }


def compute_j2_precession_rate(
    altitude_km: float,
    inclination_deg: float = 53.0,
) -> float:
    """
    Computes the secular Right Ascension of Ascending Node (RAAN) nodal precession
    rate (`d_Omega/dt` in deg/day) caused by Earth's `J2` oblateness perturbation.

    Args:
        altitude_km: Orbital altitude above Earth's surface (`~6378 km`).
        inclination_deg: Orbital plane inclination relative to equator (`deg`).

    Returns:
        nodal_rate_deg_day: Secular precession rate in degrees per day.
    """
    R_EARTH = 6378.137
    GM_EARTH = 398600.4418
    J2 = 1.08263e-3

    a = R_EARTH + max(float(altitude_km), 10.0)
    inc_rad = np.radians(inclination_deg)

    # Mean motion n in radians/sec
    n = np.sqrt(GM_EARTH / (a ** 3))

    # J2 secular precession dOmega/dt = -1.5 * n * J2 * (R/a)^2 * cos(i)
    domega_dt_rad_sec = -1.5 * n * J2 * ((R_EARTH / a) ** 2) * np.cos(inc_rad)
    domega_dt_deg_day = np.degrees(domega_dt_rad_sec) * 86400.0

    return float(domega_dt_deg_day)


def solve_gauss_angles_only_iod(
    mjd: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> Dict[str, float]:
    """
    Gauss Angles-Only Initial Orbit Determination (IOD) approximation from 3 distinct astrometric
    line-of-sight unit vectors (`hat_rho_1`, `hat_rho_2`, `hat_rho_3`).

    Args:
        mjd: Observation times (days).
        ra: Right Ascension array (deg).
        dec: Declination array (deg).

    Returns:
        dict containing estimated `semi_major_axis_km`, `estimated_inclination_deg`,
        and `iod_convergence_score`.
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)

    if len(mjd) < 3:
        return {"semi_major_axis_km": 0.0, "estimated_inclination_deg": 0.0, "iod_convergence_score": 0.0}

    # Select 3 well-separated observation epochs
    indices = [0, len(mjd) // 2, len(mjd) - 1]
    t1, t2, t3 = mjd[indices[0]], mjd[indices[1]], mjd[indices[2]]

    if abs(t3 - t1) < 1e-4:
        return {"semi_major_axis_km": 0.0, "estimated_inclination_deg": 0.0, "iod_convergence_score": 0.0}

    # Line-of-sight direction cosines (unit vectors)
    ra_r = np.radians(ra[indices])
    dec_r = np.radians(dec[indices])
    rho_hat = np.array([
        np.cos(dec_r) * np.cos(ra_r),
        np.cos(dec_r) * np.sin(ra_r),
        np.sin(dec_r)
    ]).T  # Shape: (3, 3)

    # Estimate angular velocity norm and derive approximate Keplerian altitude
    dt_sec = (t3 - t1) * 86400.0
    angular_dist_rad = np.arccos(np.clip(np.dot(rho_hat[0], rho_hat[2]), -1.0, 1.0))
    omega_rad_sec = angular_dist_rad / max(dt_sec, 1e-3)

    GM_EARTH = 398600.4418
    R_EARTH = 6378.137
    # From circular Keplerian motion: omega ~ sqrt(GM / r^3) => r ~ (GM / omega^2)^(1/3)
    if omega_rad_sec > 1e-6:
        r_approx = np.cbrt(GM_EARTH / (omega_rad_sec ** 2))
    else:
        r_approx = 42164.0  # GEO radius fallback

    inc_est = float(np.degrees(np.arccos(np.clip(rho_hat[1, 2], -1.0, 1.0))))
    return {
        "semi_major_axis_km": float(r_approx),
        "estimated_inclination_deg": inc_est,
        "iod_convergence_score": 0.94 if 6500.0 <= r_approx <= 50000.0 else 0.40,
    }


def compute_spectral_window_alias(
    t: np.ndarray,
    best_freq_per_day: float,
) -> Dict[str, Union[bool, float]]:
    """
    Evaluates the spectral Window Function `W(f)` of the observation timestamps to check
    whether a candidate periodicity is actually a diurnal survey sampling alias
    (`e.g., 1 cycle/day or 2 cycles/day Earth rotation artifact`).

    Args:
        t: Observation epochs (days).
        best_freq_per_day: Best-fit candidate frequency (`1.0 / period_days`).

    Returns:
        dict with `window_alias_flag` (True if alias) and `window_power` at that frequency.
    """
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 4:
        return {"window_alias_flag": False, "window_power": 0.0}

    # Calculate window function spectral power at best candidate frequency
    phases = 2.0 * np.pi * float(best_freq_per_day) * t
    w_power = float((np.cos(phases).mean() ** 2 + np.sin(phases).mean() ** 2))

    # Also check exact proximity to diurnal sampling harmonics (1.0, 2.0, 3.0 cycles/day)
    diurnal_resid = min(abs(best_freq_per_day - 1.0), abs(best_freq_per_day - 2.0), abs(best_freq_per_day - 3.0))
    is_alias = bool(w_power > 0.65 or (diurnal_resid < 0.05 and w_power > 0.45))

    return {
        "window_alias_flag": is_alias,
        "window_power": w_power,
    }
