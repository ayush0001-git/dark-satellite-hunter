"""
Supervised and Unsupervised Baselines (`baselines.py`)

1. `DomainFeatureExtractor` - hand-crafted astrophysical variability features
   (amplitudes, color, skew/kurtosis, Lomb-Scargle period, reduced chi^2,
   von Neumann eta, glint spike ratio).
2. `FastMiniRocketKernels` / `MiniRocketBaseline` - simplified reimplementation
   of MiniRocket (Dempster et al. 2021): random dilated +/-1 kernels with
   PPV pooling, classified by RidgeClassifierCV.
3. `evaluate_baselines` - benchmarks both on a stratified held-out split.
"""
import numpy as np
import scipy.stats as stats
from scipy.signal import lombscargle
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional

N_DOMAIN_FEATURES = 18


class DomainFeatureExtractor:
    """
    Physically motivated statistical features from irregular multi-band
    light curves. Uses only (t, mag, magerr, band) - never ground-truth
    labels - so it is safe for unsupervised discovery.
    """

    def __init__(self, freqs: Optional[np.ndarray] = None, n_freqs: int = 800):
        if freqs is None:
            # Log-spaced angular frequency grid covering periods 0.05 - 50 d.
            periods = np.logspace(np.log10(0.05), np.log10(50.0), n_freqs)
            self.freqs = 2.0 * np.pi / periods
        else:
            self.freqs = freqs

    def extract_single(
        self,
        t: np.ndarray,
        mag: np.ndarray,
        band: np.ndarray,
        magerr: Optional[np.ndarray] = None,
        ra: Optional[np.ndarray] = None,
        dec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Feature vector of length 18:
             0: g-band amplitude (p95 - p05)
             1: r-band amplitude (p95 - p05)
             2: g - r median color difference
             3: g-band skewness (negative skew = bright spikes)
             4: r-band skewness
             5: g-band excess kurtosis
             6: r-band excess kurtosis
             7: log10 dominant period (days) via Lomb-Scargle
             8: periodogram peak / median power ratio
             9: glint spike ratio (fraction of epochs > 3 robust sigma bright of median)
            10: log10 reduced chi^2 vs a constant-source model
            11: von Neumann eta (mean successive difference / variance)
            12: prewhitened glint ratio - bright outliers AFTER subtracting the
                best-fit sinusoid. Smooth pulsators (RR Lyrae) whiten away;
                stochastic specular glints survive. Key debris discriminator.
            13: prewhitened maximum bright deviation in robust-sigma units.
            14: chromatic glint ratio - log of (r-band : g-band) prewhitened
                bright-outlier strength. Surface-material signature: Kapton/MLI
                glints are red-dominant (+), solar-cell glints blue-dominant (-),
                astrophysical residuals achromatic (~0).
            15: astrometry-photometry correlation - |Pearson r| between bright
                deviation and the signed projection of detrended positional
                residuals onto their principal axis. Blended/moving reflectors
                show centroid wobble locked to brightness; stars do not.
            16: detrended astrometric scatter (arcsec, robust sigma of the
                positional residuals after removing linear proper motion).
            17: one-sided outlier asymmetry - prewhitened bright-outlier
                fraction minus faint-outlier fraction (glints +, eclipses -).

        Features 15-16 are 0 when `ra`/`dec` are unavailable, so purely
        photometric archives degrade gracefully.
        """
        f = np.zeros(N_DOMAIN_FEATURES, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        mag = np.asarray(mag, dtype=np.float64)
        band = np.asarray(band)

        g_mask = band == 0
        r_mask = band == 1

        g_med = r_med = np.median(mag) if len(mag) else 0.0
        if np.sum(g_mask) >= 5:
            g = mag[g_mask]
            f[0] = np.percentile(g, 95) - np.percentile(g, 5)
            f[3] = stats.skew(g)
            f[5] = stats.kurtosis(g)
            g_med = np.median(g)
        if np.sum(r_mask) >= 5:
            r = mag[r_mask]
            f[1] = np.percentile(r, 95) - np.percentile(r, 5)
            f[4] = stats.skew(r)
            f[6] = stats.kurtosis(r)
            r_med = np.median(r)
        f[2] = g_med - r_med

        # Lomb-Scargle dominant period.
        best_omega = None
        if len(t) >= 10:
            try:
                pgram = lombscargle(t - t[0], mag - np.median(mag), self.freqs, normalize=True)
                best = int(np.argmax(pgram))
                best_omega = float(self.freqs[best])
                f[7] = np.log10(max(2.0 * np.pi / best_omega, 1e-3))
                f[8] = pgram[best] / (np.median(pgram) + 1e-6)
            except Exception:
                f[7], f[8] = 0.0, 1.0

        if len(mag) >= 10:
            med = np.median(mag)
            # Robust sigma from MAD so the glints themselves don't inflate it.
            robust_sigma = 1.4826 * np.median(np.abs(mag - med)) + 1e-6
            f[9] = np.mean((med - mag) > 3.0 * robust_sigma)  # bright outliers only

            # Reduced chi^2 of a constant-brightness model: the classical
            # "is this source variable at all?" statistic.
            if magerr is not None and len(magerr) == len(mag):
                err = np.clip(np.asarray(magerr, dtype=np.float64), 1e-3, None)
                chi2_nu = np.sum(((mag - med) / err) ** 2) / max(len(mag) - 1, 1)
                f[10] = np.log10(max(chi2_nu, 1e-3))

            # von Neumann eta: low for smooth/correlated variability,
            # ~2 for white noise, distinguishes trends from jitter.
            order = np.argsort(t)
            m_sorted = mag[order]
            var = np.var(m_sorted) + 1e-9
            f[11] = np.mean(np.diff(m_sorted) ** 2) / var

            # Prewhitened glint statistics: subtract the best-fit sinusoid at
            # the dominant Lomb-Scargle frequency (linear least squares on
            # sin/cos + constant), then look for bright outliers that the
            # periodic model cannot absorb.
            residual = mag - med
            if best_omega is not None:
                design = np.column_stack([
                    np.sin(best_omega * (t - t[0])),
                    np.cos(best_omega * (t - t[0])),
                    np.ones(len(t)),
                ])
                try:
                    coeffs, *_ = np.linalg.lstsq(design, mag, rcond=None)
                    residual = mag - design @ coeffs
                except np.linalg.LinAlgError:
                    pass
            res_med = np.median(residual)
            res_sigma = 1.4826 * np.median(np.abs(residual - res_med)) + 1e-6
            bright_res = (res_med - residual) / res_sigma
            faint_res = (residual - res_med) / res_sigma
            f[12] = np.mean(bright_res > 3.0)
            f[13] = np.max(bright_res)

            # 14: chromatic glint ratio (material signature). Compare the
            # strength of prewhitened bright excursions between bands.
            if np.sum(g_mask) >= 5 and np.sum(r_mask) >= 5:
                s_g = max(np.percentile(bright_res[g_mask], 98), 0.0)
                s_r = max(np.percentile(bright_res[r_mask], 98), 0.0)
                f[14] = np.log((s_r + 0.5) / (s_g + 0.5))

            # 15-16: astrometric wobble statistics.
            if ra is not None and dec is not None and len(ra) == len(t):
                ra = np.asarray(ra, dtype=np.float64)
                dec = np.asarray(dec, dtype=np.float64)
                cosd = max(np.cos(np.radians(np.median(dec))), 1e-6)
                tt = t - t[0]
                # Remove linear proper motion so only the wobble remains.
                res_ra = (ra - np.polyval(np.polyfit(tt, ra, 1), tt)) * cosd * 3600.0
                res_dec = (dec - np.polyval(np.polyfit(tt, dec, 1), tt)) * 3600.0
                pos = np.column_stack([res_ra, res_dec])
                pos -= pos.mean(axis=0)
                # Signed wobble coordinate: projection onto the principal axis.
                try:
                    _, _, vt = np.linalg.svd(pos, full_matrices=False)
                    wobble = pos @ vt[0]
                except np.linalg.LinAlgError:
                    wobble = res_ra
                if np.std(wobble) > 1e-9 and np.std(bright_res) > 1e-9:
                    f[15] = abs(np.corrcoef(bright_res, wobble)[0, 1])
                disp = np.hypot(res_ra, res_dec)
                f[16] = 1.4826 * np.median(np.abs(disp - np.median(disp)))

            # 17: one-sided outlier asymmetry (glints vs eclipses).
            f[17] = np.mean(bright_res > 3.0) - np.mean(faint_res > 3.0)

        return f.astype(np.float32)

    def fit_transform(self, records: List[Dict[str, np.ndarray]]) -> np.ndarray:
        return np.vstack([
            self.extract_single(
                rec["t"], rec["mag"], rec["band"], rec.get("magerr"),
                ra=rec.get("ra"), dec=rec.get("dec"),
            )
            for rec in records
        ])


def infer_glint_material(
    chromatic_log_ratio: float,
    num_glints: int,
    min_glints: int = 2,
    red_threshold: float = 0.12,
    blue_threshold: float = -0.10,
) -> str:
    """
    Data-only surface-material hypothesis from glint chromaticity (feature 14).

    Kapton/MLI thermal blankets reflect red more strongly (r-dominant glints);
    silicon solar cells reflect blue more strongly at high incidence angles
    (g-dominant glints). With too few glints the chromatic ratio is noise, so
    no inference is made. This is a coarse hypothesis generator for follow-up
    spectroscopy, not a classification.
    """
    if num_glints < min_glints:
        return "INSUFFICIENT_GLINTS"
    if chromatic_log_ratio >= red_threshold:
        return "MLI_KAPTON_LIKE (red-dominant glints)"
    if chromatic_log_ratio <= blue_threshold:
        return "SOLAR_CELL_LIKE (blue-dominant glints)"
    return "NEUTRAL_OR_ALUMINUM"


class FastMiniRocketKernels:
    """
    Simplified MiniRocket-style transform: `num_kernels` random dilated
    kernels of length 9 with +/-1 weights, pooled by PPV (proportion of
    positive values). Self-contained (no sktime dependency).
    """

    def __init__(self, num_kernels: int = 1000, seq_len: int = 256, in_channels: int = 2, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.num_kernels = num_kernels
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.dilations = rng.choice([1, 2, 4, 8, 16], size=num_kernels)
        self.weights = rng.choice([-1.0, 1.0], size=(num_kernels, in_channels, 9))
        self.biases = rng.normal(0.0, 1.0, size=num_kernels)

    def transform_batch(self, x: np.ndarray) -> np.ndarray:
        """x: [B, C, L]  ->  PPV features [B, num_kernels]."""
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.num_kernels), dtype=np.float32)

        for k in range(self.num_kernels):
            dil = int(self.dilations[k])
            rf = (9 - 1) * dil + 1
            if rf >= self.seq_len:
                out[:, k] = 0.5
                continue
            width = self.seq_len - rf + 1
            conv = np.zeros((batch_size, width), dtype=np.float32)
            for ch in range(self.in_channels):
                for i in range(9):
                    conv += x[:, ch, i * dil : i * dil + width] * self.weights[k, ch, i]
            out[:, k] = np.mean(conv > self.biases[k], axis=1)
        return out


class MiniRocketBaseline:
    """MiniRocket-style kernels + RidgeClassifierCV."""

    def __init__(self, num_kernels: int = 1000, seq_len: int = 256, seed: int = 42):
        self.kernels = FastMiniRocketKernels(num_kernels=num_kernels, seq_len=seq_len, seed=seed)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MiniRocketBaseline":
        self.classifier.fit(self.kernels.transform_batch(X), y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.decision_function(self.kernels.transform_batch(X))


def _records_to_multichannel(records: List[Dict], seq_len: int = 256) -> np.ndarray:
    """Packs records into the [B, 2, seq_len] tensor the kernels expect."""
    X = np.zeros((len(records), 2, seq_len), dtype=np.float32)
    for i, rec in enumerate(records):
        mag, band = np.asarray(rec["mag"]), np.asarray(rec["band"])
        L = min(len(band), seq_len)
        g = band[:L] == 0
        r = band[:L] == 1
        X[i, 0, :L][g] = mag[:L][g]
        X[i, 1, :L][r] = mag[:L][r]
    return X


def evaluate_baselines(
    records: List[Dict[str, np.ndarray]],
    test_size: float = 0.3,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Benchmarks an unsupervised Isolation Forest (on domain features) and a
    supervised MiniRocket+Ridge classifier, both evaluated on the same
    stratified held-out split so numbers are comparable and honest.
    """
    labels = np.array([int(rec.get("label", 0)) for rec in records])
    if len(np.unique(labels)) < 2:
        return {"error": "Dataset needs both clean (0) and debris (1) samples to compute AUC."}

    idx_train, idx_test = train_test_split(
        np.arange(len(records)), test_size=test_size, stratify=labels, random_state=seed
    )
    y_train, y_test = labels[idx_train], labels[idx_test]

    # ---- 1. Unsupervised: Isolation Forest on domain features ------------
    extractor = DomainFeatureExtractor()
    feats = extractor.fit_transform(records)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    iso.fit(feats[idx_train])                       # unsupervised: labels unused
    iso_scores = -iso.decision_function(feats[idx_test])

    iso_roc = roc_auc_score(y_test, iso_scores)
    iso_pr = average_precision_score(y_test, iso_scores)

    clean_scores = np.sort(iso_scores[y_test == 0])
    if len(clean_scores) > 0:
        thresh = clean_scores[min(int(np.ceil(0.99 * len(clean_scores))), len(clean_scores) - 1)]
        iso_recall_1fpr = float(np.mean(iso_scores[y_test == 1] >= thresh)) if np.any(y_test == 1) else 0.0
    else:
        iso_recall_1fpr = 0.0

    # ---- 2. Supervised: MiniRocket + Ridge -------------------------------
    X_mc = _records_to_multichannel(records)
    if len(np.unique(y_train)) > 1:
        mr = MiniRocketBaseline(num_kernels=500, seed=seed).fit(X_mc[idx_train], y_train)
        mr_scores = mr.predict_scores(X_mc[idx_test])
        mr_roc = roc_auc_score(y_test, mr_scores)
        mr_pr = average_precision_score(y_test, mr_scores)
    else:
        mr_roc = mr_pr = 0.5

    return {
        "IsolationForest_ROC_AUC": float(iso_roc),
        "IsolationForest_PR_AUC": float(iso_pr),
        "IsolationForest_Recall@1%FPR": float(iso_recall_1fpr),
        "MiniRocket_ROC_AUC": float(mr_roc),
        "MiniRocket_PR_AUC": float(mr_pr),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
    }
