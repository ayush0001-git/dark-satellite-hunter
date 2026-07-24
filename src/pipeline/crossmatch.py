"""
Automated Cross-Match Engine (`crossmatch.py`)

Tags candidate anomalies against known-object catalogs using ONLY observable
information (sky position, epoch). Sources of truth, in priority order:

1. Reference star/QSO catalog     - positional cone search (vectorized).
   In mock mode this is the simulation's catalog of cataloged sources; on
   real data pass a Gaia/Simbad extract with the same schema.
2. Space-Track TLEs via SGP4      - each TLE is propagated to the observation
   epoch, converted to topocentric RA/Dec as seen from Palomar (ZTF), and
   cone-matched. TLE positional accuracy at GEO is ~tens of arcsec, so the
   default satellite cone radius is generous (60").
3. Minor Planet Center / Simbad   - optional online cone searches through
   astroquery (disabled in offline mode).

Anything unmatched with a surprisal score above `anomaly_threshold` becomes
an UNCORRELATED_ANOMALY; unmatched low-score objects are tagged as low
significance. Ground-truth simulation labels are NEVER consulted here.
"""
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ZTF / Palomar Observatory (P48)
PALOMAR_LAT_DEG = 33.3563
PALOMAR_LON_DEG = -116.8650
PALOMAR_ALT_KM = 1.712

# WGS-84 ellipsoid
_EARTH_A_KM = 6378.137
_EARTH_F = 1.0 / 298.257223563


def angular_separation_arcsec(
    ra1_deg: float, dec1_deg: float, ra2_deg: np.ndarray, dec2_deg: np.ndarray
) -> np.ndarray:
    """Great-circle separation (arcsec) between one point and an array of points."""
    ra1, dec1 = np.radians(ra1_deg), np.radians(dec1_deg)
    ra2, dec2 = np.radians(np.asarray(ra2_deg)), np.radians(np.asarray(dec2_deg))
    # Vincenty formula: numerically stable at all separations.
    dra = ra2 - ra1
    num = np.hypot(
        np.cos(dec2) * np.sin(dra),
        np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra),
    )
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.degrees(np.arctan2(num, den)) * 3600.0


def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (IAU 1982 approximation)."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    ) % 360.0
    return math.radians(gmst_deg)


def _observer_eci_km(jd_ut1: float, lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """Observer geodetic position -> ECI (equinox-of-date, adequate for TEME tagging)."""
    lat = math.radians(lat_deg)
    lst = _gmst_rad(jd_ut1) + math.radians(lon_deg)  # local sidereal angle
    e2 = _EARTH_F * (2.0 - _EARTH_F)
    n = _EARTH_A_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    r_xy = (n + alt_km) * math.cos(lat)
    return np.array([
        r_xy * math.cos(lst),
        r_xy * math.sin(lst),
        (n * (1.0 - e2) + alt_km) * math.sin(lat),
    ])


def teme_to_topocentric_radec(
    r_teme_km: np.ndarray,
    jd_ut1: float,
    lat_deg: float = PALOMAR_LAT_DEG,
    lon_deg: float = PALOMAR_LON_DEG,
    alt_km: float = PALOMAR_ALT_KM,
) -> Tuple[float, float]:
    """
    Topocentric RA/Dec (deg) of a satellite ECI(TEME) position as seen by a
    ground observer. TEME->ICRS frame rotation (precession/nutation, tens of
    arcsec) is neglected; acceptable against the 60" satellite cone radius.
    """
    rho = np.asarray(r_teme_km, dtype=np.float64) - _observer_eci_km(jd_ut1, lat_deg, lon_deg, alt_km)
    ra = math.degrees(math.atan2(rho[1], rho[0])) % 360.0
    dec = math.degrees(math.asin(rho[2] / np.linalg.norm(rho)))
    return ra, dec


class CrossmatchEngine:
    """
    Position-based cross-matching of anomaly candidates against reference
    catalogs, TLE sets, and (optionally) online services.
    """

    def __init__(
        self,
        reference_catalog: Optional[List[Dict]] = None,
        tle_path: Optional[str] = None,
        offline_mode: bool = True,
        star_cone_radius_arcsec: float = 2.0,
        sat_cone_radius_arcsec: float = 60.0,
        anomaly_threshold: float = 2.0,
    ):
        self.offline_mode = offline_mode
        self.star_cone_radius_arcsec = star_cone_radius_arcsec
        self.sat_cone_radius_arcsec = sat_cone_radius_arcsec
        self.anomaly_threshold = anomaly_threshold
        self.cache: Dict[str, Dict[str, Any]] = {}

        # Reference catalog as flat arrays for vectorized cone searches.
        self._cat_ids: List[str] = []
        self._cat_classes: List[str] = []
        self._cat_ra = np.empty(0)
        self._cat_dec = np.empty(0)
        if reference_catalog:
            self._cat_ids = [str(c["catalog_id"]) for c in reference_catalog]
            self._cat_classes = [str(c.get("obj_class", "Star")) for c in reference_catalog]
            self._cat_ra = np.array([float(c["ra_deg"]) for c in reference_catalog])
            self._cat_dec = np.array([float(c["dec_deg"]) for c in reference_catalog])

        self.tle_list: List[Tuple[str, str, str]] = []
        if tle_path:
            self._load_tles(tle_path)

    # ------------------------------------------------------------------ #
    # Catalog loading
    # ------------------------------------------------------------------ #
    def _load_tles(self, tle_path: str) -> None:
        """Loads TLEs in either 3-line (name + 2 lines) or bare 2-line format."""
        try:
            with open(tle_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip() for ln in f if ln.strip()]
        except OSError as exc:
            print(f"[crossmatch] Could not read TLE file {tle_path!r}: {exc}")
            return

        i = 0
        while i < len(lines) - 1:
            if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                # Bare 2-line format.
                self.tle_list.append((f"SAT_{len(self.tle_list):05d}", lines[i], lines[i + 1]))
                i += 2
            elif (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")
            ):
                # 3-line format: name line followed by the two element lines.
                self.tle_list.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
                i += 3
            else:
                i += 1

    # ------------------------------------------------------------------ #
    # Individual catalog checks (position/epoch only - no truth labels)
    # ------------------------------------------------------------------ #
    def check_reference_catalog(self, ra_deg: float, dec_deg: float) -> Tuple[bool, Optional[str], float]:
        """Cone search against the loaded star/QSO catalog."""
        if len(self._cat_ra) == 0:
            return False, None, float("inf")
        sep = angular_separation_arcsec(ra_deg, dec_deg, self._cat_ra, self._cat_dec)
        j = int(np.argmin(sep))
        if sep[j] <= self.star_cone_radius_arcsec:
            return True, f"{self._cat_ids[j]} ({self._cat_classes[j]})", float(sep[j])
        return False, None, float(sep[j])

    def check_spacetrack_sgp4(
        self, ra_deg: float, dec_deg: float, mjd_obs: float
    ) -> Tuple[bool, Optional[str], float]:
        """Propagates every loaded TLE to the epoch and cone-matches topocentrically."""
        if not self.tle_list:
            return False, None, float("inf")
        try:
            from sgp4.api import Satrec
        except ImportError:
            print("[crossmatch] sgp4 not installed - skipping TLE check.")
            return False, None, float("inf")

        jd = mjd_obs + 2400000.5
        jd_int, jd_frac = divmod(jd, 1.0)
        best_name, best_sep = None, float("inf")

        for name, l1, l2 in self.tle_list:
            try:
                sat = Satrec.twoline2rv(l1, l2)
                err, r, _ = sat.sgp4(jd_int, jd_frac)
                if err != 0:
                    continue
                sat_ra, sat_dec = teme_to_topocentric_radec(np.array(r), jd)
                sep = float(angular_separation_arcsec(ra_deg, dec_deg, np.array([sat_ra]), np.array([sat_dec]))[0])
                if sep < best_sep:
                    best_name, best_sep = name, sep
            except Exception:
                continue

        if best_sep <= self.sat_cone_radius_arcsec:
            return True, best_name, best_sep
        return False, None, best_sep

    def check_mpc_asteroids(self, ra_deg: float, dec_deg: float, mjd_obs: float,
                            cone_radius_arcmin: float = 1.0) -> Tuple[bool, Optional[str]]:
        """Online MPChecker-style query via astroquery (skipped offline)."""
        if self.offline_mode:
            return False, None
        try:
            from astropy.coordinates import SkyCoord
            from astropy.time import Time
            import astropy.units as u
            from astroquery.imcce import Skybot
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            result = Skybot.cone_search(coord, cone_radius_arcmin * u.arcmin,
                                        Time(mjd_obs, format="mjd"))
            if result is not None and len(result) > 0:
                return True, str(result["Name"][0])
        except Exception:
            pass
        return False, None

    def check_simbad(self, ra_deg: float, dec_deg: float,
                     cone_radius_arcsec: float = 3.0) -> Tuple[bool, Optional[str]]:
        """Online Simbad cone search via astroquery (skipped offline)."""
        if self.offline_mode:
            return False, None
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            from astroquery.simbad import Simbad
            coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            table = Simbad.query_region(coord, radius=cone_radius_arcsec * u.arcsec)
            if table is not None and len(table) > 0:
                col = "MAIN_ID" if "MAIN_ID" in table.colnames else table.colnames[0]
                return True, str(table[col][0])
        except Exception:
            pass
        return False, None

    # ------------------------------------------------------------------ #
    # Final tagging
    # ------------------------------------------------------------------ #
    def tag_candidate(
        self,
        object_id: str,
        ra_deg: float,
        dec_deg: float,
        mjd_obs: float = 59100.0,
        surprisal_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Full cross-match cascade for one candidate. Uses position, epoch and
        the data-derived surprisal score only.
        """
        if object_id in self.cache:
            return self.cache[object_id]

        def _finish(status: str, matched: str, confidence: str, sep: float = float("nan")) -> Dict[str, Any]:
            res = {
                "object_id": object_id,
                "status": status,
                "matched_catalog": matched,
                "match_separation_arcsec": None if np.isnan(sep) else round(sep, 2),
                "confidence": confidence,
            }
            self.cache[object_id] = res
            return res

        # 1. Known star/QSO by position (local, fast).
        hit, name, sep = self.check_reference_catalog(ra_deg, dec_deg)
        if hit:
            return _finish("KNOWN_STAR_OR_QSO", name, "HIGH", sep)

        # 2. Known satellite via TLE + SGP4 (local physics - works offline).
        hit, name, sep = self.check_spacetrack_sgp4(ra_deg, dec_deg, mjd_obs)
        if hit:
            return _finish("KNOWN_SATELLITE", f"SpaceTrack_{name}", "HIGH", sep)

        # 3. Below the anomaly threshold nothing changes scientifically, so
        #    skip the (slow, rate-limited) online services entirely.
        if surprisal_score < self.anomaly_threshold:
            return _finish("UNMATCHED_LOW_SIGNIFICANCE", "NONE", "LOW")

        # 4. High-surprisal candidates get the full online vetting cascade
        #    (only when offline_mode=False and astroquery is available).
        hit, name = self.check_mpc_asteroids(ra_deg, dec_deg, mjd_obs)
        if hit:
            return _finish("KNOWN_ASTEROID", f"MPC_{name}", "HIGH")

        hit, name = self.check_simbad(ra_deg, dec_deg)
        if hit:
            return _finish("KNOWN_STAR_OR_QSO", f"SIMBAD_{name}", "MEDIUM")

        # 5. Survived every catalog: an uncorrelated anomaly.
        conf = "VERY_HIGH" if surprisal_score >= 2.0 * self.anomaly_threshold else "HIGH"
        return _finish("UNCORRELATED_ANOMALY", "NONE", conf)
