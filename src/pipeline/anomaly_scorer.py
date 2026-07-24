"""
Hybrid Anomaly Scorer (`anomaly_scorer.py`)

Surprisal = alpha * z(reconstruction error) + beta * z(latent-space distance)

* z-scores use robust location/scale (median, IQR/1.349) estimated on a
  reference population of "normal" objects, so a handful of contaminating
  anomalies in the reference set cannot poison the normalization.
* Latent density supports Mahalanobis distance (Ledoit-Wolf shrinkage
  covariance - essential when d_model is comparable to the reference sample
  size), k-NN mean distance, or Isolation Forest.
"""
import numpy as np
import scipy.linalg as linalg
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional

_IQR_TO_SIGMA = 1.349  # IQR of a Gaussian in units of sigma


def _robust_loc_scale(values: np.ndarray) -> Dict[str, float]:
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    scale = float((q75 - q25) / _IQR_TO_SIGMA)
    if scale < 1e-9:
        scale = float(np.std(values)) + 1e-9
    return {"loc": med, "scale": scale}


class HybridAnomalyScorer:
    """
    Combines reconstruction-error and latent-density evidence into a single
    surprisal score, calibrated against a reference (assumed-normal) sample.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        density_method: str = "mahalanobis",   # 'mahalanobis' | 'knn' | 'isolation_forest'
        n_neighbors: int = 15,
        reg_covar: float = 1e-4,
    ):
        if density_method.lower() not in ("mahalanobis", "knn", "isolation_forest"):
            raise ValueError(f"Unknown density_method: {density_method!r}")
        self.alpha = alpha
        self.beta = beta
        self.density_method = density_method.lower()
        self.n_neighbors = n_neighbors
        self.reg_covar = reg_covar

        self._fitted = False
        self.recon_stats = {"loc": 0.0, "scale": 1.0}
        self.latent_stats = {"loc": 0.0, "scale": 1.0}
        self.latent_mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.knn_model: Optional[NearestNeighbors] = None
        self.iso_model: Optional[IsolationForest] = None
        # Robust per-dimension standardization fitted on the reference set,
        # so heavy-tailed dimensions cannot dominate the density estimate.
        self._dim_loc: Optional[np.ndarray] = None
        self._dim_scale: Optional[np.ndarray] = None

    def _standardize(self, latents: np.ndarray) -> np.ndarray:
        if self._dim_loc is None or self._dim_scale is None:
            return latents
        return (latents - self._dim_loc) / self._dim_scale

    def fit(self, reference_latents: np.ndarray, reference_recon_errors: np.ndarray) -> "HybridAnomalyScorer":
        """
        Calibrates the scorer on a reference population (e.g. objects that
        cross-matched to known catalog sources).
        """
        reference_latents = np.asarray(reference_latents, dtype=np.float64)
        reference_recon_errors = np.asarray(reference_recon_errors, dtype=np.float64)
        if reference_latents.ndim != 2 or len(reference_latents) < 3:
            raise ValueError("Need at least 3 reference objects with 2-D latent array.")

        self.recon_stats = _robust_loc_scale(reference_recon_errors)

        # Fit robust per-dimension scales on the reference population.
        self._dim_loc = np.median(reference_latents, axis=0)
        q75, q25 = np.percentile(reference_latents, [75, 25], axis=0)
        scale = (q75 - q25) / _IQR_TO_SIGMA
        std_fallback = reference_latents.std(axis=0)
        scale = np.where(scale < 1e-9, std_fallback, scale)
        self._dim_scale = np.where(scale < 1e-9, 1.0, scale)

        ref_std = self._standardize(reference_latents)
        self.latent_mean = ref_std.mean(axis=0)
        N, D = ref_std.shape

        if self.density_method == "mahalanobis":
            centered = ref_std - self.latent_mean
            try:
                # Shrinkage covariance: well-conditioned even when N ~ D.
                cov = LedoitWolf().fit(ref_std).covariance_
            except Exception:
                cov = np.cov(centered, rowvar=False)
            cov = cov + np.eye(D) * self.reg_covar
            try:
                self.cov_inv = linalg.pinvh(cov)
            except Exception:
                self.cov_inv = np.eye(D) / self.reg_covar

        elif self.density_method == "knn":
            k = min(self.n_neighbors, N - 1) if N > 1 else 1
            self.knn_model = NearestNeighbors(n_neighbors=k).fit(ref_std)

        else:  # isolation_forest
            self.iso_model = IsolationForest(
                n_estimators=300, contamination="auto", random_state=42
            ).fit(ref_std)

        ref_dists = self.compute_latent_distances(reference_latents)
        self.latent_stats = _robust_loc_scale(ref_dists)
        self._fitted = True
        return self

    def compute_latent_distances(self, latents: np.ndarray) -> np.ndarray:
        """Raw latent-space anomaly distances for the fitted density model."""
        # Apply the same robust per-dimension standardization the reference
        # population was fitted with - fit/score asymmetry here silently
        # destroys the density estimate.
        latents = self._standardize(np.asarray(latents, dtype=np.float64))

        if self.density_method == "mahalanobis":
            if self.latent_mean is None or self.cov_inv is None:
                return np.zeros(len(latents))
            diff = latents - self.latent_mean
            return np.sqrt(np.maximum(np.sum((diff @ self.cov_inv) * diff, axis=1), 0.0))

        if self.density_method == "knn":
            if self.knn_model is None:
                return np.zeros(len(latents))
            dists, _ = self.knn_model.kneighbors(latents)
            return dists.mean(axis=1)

        if self.iso_model is None:
            return np.zeros(len(latents))
        return -self.iso_model.decision_function(latents)

    def score(self, recon_errors: np.ndarray, latents: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Hybrid surprisal scores for a set of objects.

        Returns dict with:
            surprisal_score, recon_zscore, latent_zscore,
            raw_recon, raw_latent_dist
        """
        if not self._fitted:
            raise RuntimeError("HybridAnomalyScorer.score() called before fit().")

        recon_errors = np.asarray(recon_errors, dtype=np.float64)
        recon_z = (recon_errors - self.recon_stats["loc"]) / self.recon_stats["scale"]

        raw_dist = self.compute_latent_distances(latents)
        latent_z = (raw_dist - self.latent_stats["loc"]) / self.latent_stats["scale"]

        return {
            "surprisal_score": self.alpha * recon_z + self.beta * latent_z,
            "recon_zscore": recon_z,
            "latent_zscore": latent_z,
            "raw_recon": recon_errors,
            "raw_latent_dist": raw_dist,
        }
