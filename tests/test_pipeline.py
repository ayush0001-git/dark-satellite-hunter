"""
Fast unit/integration tests for the Dark Satellite Hunter pipeline.

Run with:  python -m pytest tests/ -q
"""
import copy
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.mock_generator import build_reference_catalog, generate_mock_ztf_dataset
from src.data.synthetic_debris import inject_debris_into_lightcurve, tumbling_glint_model
from src.data.ztf_dataset import LightCurveDataset, collate_lightcurves
from src.models.baselines import DomainFeatureExtractor, evaluate_baselines
from src.models.patchtst_mae import PatchTSTMaskedAutoencoder
from src.pipeline.anomaly_scorer import HybridAnomalyScorer
from src.pipeline.crossmatch import CrossmatchEngine, angular_separation_arcsec
from src.pipeline.orbital_fit import estimate_glint_periodicity, fit_preliminary_orbit


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def test_mock_generator_reproducible():
    a = generate_mock_ztf_dataset(n_clean=20, n_debris=5, seed=7)
    b = generate_mock_ztf_dataset(n_clean=20, n_debris=5, seed=7)
    assert [r["object_id"] for r in a] == [r["object_id"] for r in b]
    np.testing.assert_allclose(a[0]["mag"], b[0]["mag"])
    np.testing.assert_allclose(a[-1]["ra"], b[-1]["ra"])


def test_mock_generator_sane_photometry():
    records = generate_mock_ztf_dataset(n_clean=30, n_debris=10, seed=3)
    labels = np.array([r["label"] for r in records])
    assert labels.sum() == 10 and (labels == 0).sum() == 30
    for rec in records:
        raw = rec["raw_mag"]
        # Astronomical magnitudes must stay physically plausible for a
        # ground-based survey (the old flux model produced mag ~ -0.5).
        assert np.all(raw > 10.0) and np.all(raw < 25.0), \
            f"{rec['object_id']} has non-physical magnitudes [{raw.min():.2f}, {raw.max():.2f}]"
        assert len(rec["t"]) == len(rec["mag"]) == len(rec["band"]) == len(rec["ra"])


def test_debris_injection_brightens_not_darkens():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 200))
    mag = np.full(200, 20.0)
    magerr = np.full(200, 0.03)
    _, new_mag, new_err, glints = inject_debris_into_lightcurve(
        t, mag, magerr, period=0.2, amplitude=1.5, glint_prob=0.1,
        debris_mean_mag=17.5, seed=1,
    )
    # Adding flux can only make the blend brighter (smaller magnitude).
    assert np.all(new_mag <= mag + 1e-9)
    assert np.all((new_err >= 0.005) & (new_err <= 0.5))
    # Glint epochs should be brighter than the median of the blend.
    if glints.any():
        assert np.median(new_mag[glints]) < np.median(new_mag)


def test_tumbling_model_zero_mean_diffuse():
    t = np.linspace(0, 10, 5000)
    delta, glints = tumbling_glint_model(t, period=0.3, amplitude=2.0, glint_prob=0.0, seed=5)
    assert not glints.any()
    assert abs(np.mean(delta)) < 0.05  # diffuse term is ~zero-mean
    assert np.max(delta) - np.min(delta) <= 2.0 * 1.3 + 1e-6


# --------------------------------------------------------------------------- #
# Dataset / model
# --------------------------------------------------------------------------- #
def _tiny_batch(n=4, seq_len=256):
    records = generate_mock_ztf_dataset(n_clean=n, n_debris=0, seed=11)
    ds = LightCurveDataset(records, seq_len=seq_len, train=False)
    return collate_lightcurves([ds[i] for i in range(len(ds))])


def test_dataset_shapes_and_obs_mask():
    batch = _tiny_batch()
    assert batch["x"].shape[1:] == (2, 256)
    assert batch["t"].shape[1:] == (256,)
    obs = batch["obs_mask"].numpy()
    x = batch["x"].numpy()
    # Padded epochs must carry zero flux in both channels.
    assert np.all(x[:, :, :][:, :, obs[0] == 0][0] == 0.0) if (obs[0] == 0).any() else True
    # Normalized time stays in [0, 1].
    assert batch["t"].min() >= 0.0 and batch["t"].max() <= 1.0


def test_model_forward_and_losses():
    torch.manual_seed(0)
    model = PatchTSTMaskedAutoencoder(seq_len=256, patch_len=16, d_model=64,
                                      enc_layers=1, enc_heads=4, dec_dim=32,
                                      dec_layers=1, dec_heads=2)
    batch = _tiny_batch()
    preds, targets, mask = model(batch["x"], batch["t"])
    B = batch["x"].shape[0]
    assert preds.shape == (B, 16, 32) and targets.shape == (B, 16, 32) and mask.shape == (B, 16)
    # 75% of the 16 patches masked -> 12 per sample.
    assert torch.all(mask.sum(dim=1) == 12)

    obs_w = model.patch_observation_weights(batch["obs_mask"], B, batch["x"].device)
    loss = model.compute_loss(preds, targets, mask, patch_weights=obs_w)
    assert torch.isfinite(loss)
    loss.backward()  # gradient path intact

    cls = model.extract_cls_latent(batch["x"], batch["t"])
    assert cls.shape == (B, 64) and torch.isfinite(cls).all()

    errs = model.per_sample_recon_error(batch["x"], batch["t"], batch["obs_mask"])
    assert errs.shape == (B,) and torch.isfinite(errs).all() and (errs >= 0).all()


def test_model_rejects_bad_config():
    with pytest.raises(ValueError):
        PatchTSTMaskedAutoencoder(seq_len=250, patch_len=16)
    with pytest.raises(ValueError):
        PatchTSTMaskedAutoencoder(mask_ratio=1.5)


# --------------------------------------------------------------------------- #
# Anomaly scorer
# --------------------------------------------------------------------------- #
def test_scorer_flags_outliers():
    rng = np.random.default_rng(0)
    ref_latents = rng.normal(0, 1, size=(200, 8))
    ref_errs = np.abs(rng.normal(1.0, 0.1, size=200))
    scorer = HybridAnomalyScorer().fit(ref_latents, ref_errs)

    inlier = scorer.score(np.array([1.0]), np.zeros((1, 8)))["surprisal_score"][0]
    outlier = scorer.score(np.array([5.0]), np.full((1, 8), 6.0))["surprisal_score"][0]
    assert outlier > inlier + 3.0


def test_scorer_requires_fit():
    with pytest.raises(RuntimeError):
        HybridAnomalyScorer().score(np.array([1.0]), np.zeros((1, 4)))


# --------------------------------------------------------------------------- #
# Orbital fit / periodicity
# --------------------------------------------------------------------------- #
def test_orbital_fit_recovers_linear_drift():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 100, 120))
    rate_ra, rate_dec = 0.008, -0.004  # deg/day
    jit = 0.3 / 3600.0
    ra = 120.0 + rate_ra * t + rng.normal(0, jit, len(t))
    dec = 35.0 + rate_dec * t + rng.normal(0, jit, len(t))
    kin = fit_preliminary_orbit(t, ra, dec)
    assert kin["dra_dt_deg_day"] == pytest.approx(rate_ra, rel=0.05)
    assert kin["ddec_dt_deg_day"] == pytest.approx(rate_dec, rel=0.05)
    assert kin["regime"].startswith("GEO_BELT_DRIFTER_CANDIDATE")


def test_orbital_fit_stationary_star():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 100, 100))
    jit = 0.3 / 3600.0
    kin = fit_preliminary_orbit(
        t, 200.0 + rng.normal(0, jit, len(t)), -10.0 + rng.normal(0, jit, len(t))
    )
    assert kin["regime"] == "SIDEREAL_STATIONARY"
    assert not kin["non_keplerian_flag"]


def test_orbital_fit_handles_ra_wrap():
    t = np.linspace(0, 50, 60)
    ra = (359.9 + 0.01 * t) % 360.0  # crosses 0/360
    dec = np.zeros_like(t)
    kin = fit_preliminary_orbit(t, ra, dec)
    assert kin["dra_dt_deg_day"] == pytest.approx(0.01, rel=0.05)


def test_glint_periodicity_recovery():
    rng = np.random.default_rng(4)
    true_period = 0.15  # days = 3.6 h
    t = np.sort(rng.uniform(0, 40, 250))
    mag = 18.0 + rng.normal(0, 0.02, len(t))
    phase = (t / true_period) % 1.0
    glint_epochs = phase < 0.05
    mag[glint_epochs] -= 2.5
    est = estimate_glint_periodicity(t, mag)
    assert est["num_glints"] >= 3
    ratio = est["period_days"] / true_period
    # Accept the fundamental or a low-order alias.
    assert min(abs(ratio - k) for k in (0.5, 1.0, 2.0)) < 0.05
    assert est["phase_sharpness"] > 0.5


# --------------------------------------------------------------------------- #
# Cross-matching
# --------------------------------------------------------------------------- #
def test_angular_separation_known_values():
    sep = angular_separation_arcsec(10.0, 0.0, np.array([10.0]), np.array([1.0]))[0]
    assert sep == pytest.approx(3600.0, rel=1e-6)  # 1 deg in declination
    sep0 = angular_separation_arcsec(45.0, 20.0, np.array([45.0]), np.array([20.0]))[0]
    assert sep0 == pytest.approx(0.0, abs=1e-9)


def test_crossmatch_uses_position_not_labels():
    records = generate_mock_ztf_dataset(n_clean=40, n_debris=8, seed=9)
    catalog = build_reference_catalog(records)
    engine = CrossmatchEngine(reference_catalog=catalog, offline_mode=True)

    for rec in records:
        hit, _, _ = engine.check_reference_catalog(
            float(np.mean(rec["ra"])), float(np.mean(rec["dec"]))
        )
        if rec["label"] == 0:
            assert hit, f"cataloged source {rec['object_id']} failed to match"
        # Debris may occasionally pass near a star, but its mean position
        # should almost never coincide within 2 arcsec.
    debris_hits = [
        engine.check_reference_catalog(float(np.mean(r["ra"])), float(np.mean(r["dec"])))[0]
        for r in records if r["label"] == 1
    ]
    assert sum(debris_hits) <= 1


def test_tag_candidate_statuses():
    records = generate_mock_ztf_dataset(n_clean=30, n_debris=5, seed=13)
    catalog = build_reference_catalog(records)
    engine = CrossmatchEngine(reference_catalog=catalog, offline_mode=True, anomaly_threshold=2.0)

    star = next(r for r in records if r["label"] == 0)
    tag = engine.tag_candidate("star", float(np.mean(star["ra"])), float(np.mean(star["dec"])),
                               surprisal_score=0.0)
    assert tag["status"] == "KNOWN_STAR_OR_QSO"

    # Far-away, unmatched positions: status decided by surprisal only.
    tag_hi = engine.tag_candidate("anom_hi", 10.0, -60.0, surprisal_score=5.0)
    tag_lo = engine.tag_candidate("anom_lo", 11.0, -61.0, surprisal_score=0.1)
    assert tag_hi["status"] == "UNCORRELATED_ANOMALY"
    assert tag_lo["status"] == "UNMATCHED_LOW_SIGNIFICANCE"


# --------------------------------------------------------------------------- #
# THE integrity test: discovery output is invariant to ground-truth labels
# --------------------------------------------------------------------------- #
def test_no_label_leakage_in_discovery_scores():
    records = generate_mock_ztf_dataset(n_clean=60, n_debris=12, seed=21)

    def run_pipeline(recs):
        catalog = build_reference_catalog(records)  # catalog fixed (external truth)
        feats = DomainFeatureExtractor().fit_transform(recs).astype(np.float64)
        recon = feats[:, 10]
        engine = CrossmatchEngine(reference_catalog=catalog, offline_mode=True)
        matched = np.array([
            engine.check_reference_catalog(float(np.mean(r["ra"])), float(np.mean(r["dec"])))[0]
            for r in recs
        ])
        ref = np.flatnonzero(matched) if matched.sum() >= 10 else np.arange(len(recs))
        scorer = HybridAnomalyScorer().fit(feats[ref], recon[ref])
        return scorer.score(recon, feats)["surprisal_score"]

    scores_true = run_pipeline(records)

    corrupted = copy.deepcopy(records)
    for r in corrupted:
        r["label"] = np.int64(1 - int(r["label"]))   # flip every label
        r["true_class"] = "SCRAMBLED"
    scores_flipped = run_pipeline(corrupted)

    np.testing.assert_allclose(scores_true, scores_flipped, rtol=0, atol=0,
                               err_msg="Discovery scores changed when labels were flipped -> leakage!")


def test_discovery_scores_actually_separate_debris():
    """End-to-end sanity: label-free scores must still detect the injections."""
    from sklearn.metrics import roc_auc_score
    records = generate_mock_ztf_dataset(n_clean=80, n_debris=20, seed=33)
    feats = DomainFeatureExtractor().fit_transform(records).astype(np.float64)
    catalog = build_reference_catalog(records)
    engine = CrossmatchEngine(reference_catalog=catalog, offline_mode=True)
    matched = np.array([
        engine.check_reference_catalog(float(np.mean(r["ra"])), float(np.mean(r["dec"])))[0]
        for r in records
    ])
    scorer = HybridAnomalyScorer().fit(feats[matched], feats[matched][:, 10])
    scores = scorer.score(feats[:, 10], feats)["surprisal_score"]
    labels = np.array([int(r["label"]) for r in records])
    auc = roc_auc_score(labels, scores)
    # Domain-feature fallback typically reaches 0.90-0.99 depending on the
    # draw; 0.85 guards against regressions without over-fitting to one seed.
    assert auc > 0.85, f"Label-free detection AUC unexpectedly low: {auc:.3f}"


# --------------------------------------------------------------------------- #
# Real-data ingestion (.csv / .parquet, column aliases, JD conversion)
# --------------------------------------------------------------------------- #
def test_csv_loader_with_aliased_columns(tmp_path):
    import pandas as pd
    from src.data.ztf_dataset import PolarsZTFDataset

    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "objid": ["OBJ_A"] * n,                       # alias of object_id
        "jd": 2459000.5 + np.sort(rng.uniform(0, 90, n)),  # JD, not MJD
        "magpsf": 17.0 + rng.normal(0, 0.05, n),      # alias of mag
        "sigmapsf": np.full(n, 0.03),                 # alias of magerr
        "fid": rng.choice([1, 2, 3], size=n),         # 3 = i-band, must drop
        "ra": np.full(n, 150.0),
        "dec": np.full(n, 20.0),
    })
    csv_path = tmp_path / "lightcurves.csv"
    df.to_csv(csv_path, index=False)

    ds = PolarsZTFDataset(str(csv_path), min_obs=5)
    records = ds.load_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["object_id"] == "OBJ_A"
    # JD auto-converted to MJD (JD 2459000.5 -> MJD 59000).
    assert 58990 < rec["raw_t"][0] < 59095
    # i-band epochs dropped: only g(0)/r(1) remain.
    assert set(np.unique(rec["band"])) <= {0, 1}
    assert len(rec["ra"]) == len(rec["t"])


def test_loader_rejects_missing_columns(tmp_path):
    import pandas as pd
    from src.data.ztf_dataset import PolarsZTFDataset
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        PolarsZTFDataset(str(tmp_path / "bad.csv"))


# --------------------------------------------------------------------------- #
# Advanced physics features: chromaticity, wobble, FAP, material inference
# --------------------------------------------------------------------------- #
def test_material_inference():
    from src.models.baselines import infer_glint_material
    assert "MLI_KAPTON" in infer_glint_material(0.4, num_glints=5)
    assert "SOLAR_CELL" in infer_glint_material(-0.3, num_glints=5)
    assert infer_glint_material(0.02, num_glints=5) == "NEUTRAL_OR_ALUMINUM"
    assert infer_glint_material(0.9, num_glints=1) == "INSUFFICIENT_GLINTS"


def test_period_false_alarm_probability():
    rng = np.random.default_rng(6)
    t = np.sort(rng.uniform(0, 40, 250))
    # Strongly periodic glints -> significant period, low FAP.
    mag = 18.0 + rng.normal(0, 0.02, len(t))
    mag[((t / 0.15) % 1.0) < 0.05] -= 2.5
    sig = estimate_glint_periodicity(t, mag)
    # Pure noise -> insignificant period, high FAP.
    noise = estimate_glint_periodicity(t, 18.0 + rng.normal(0, 0.05, len(t)))
    # The FAP is a *conservative upper bound* (asymptotic Rayleigh tail +
    # trials correction): ~11 glints over a 40-day baseline can reach ~0.1
    # at best, while phase-random noise saturates near 1. What matters is
    # the decisive separation, and that the recovered period is exact.
    assert sig["false_alarm_prob"] < 0.15
    assert noise["false_alarm_prob"] > 0.5
    assert sig["period_days"] == pytest.approx(0.15, rel=0.01)
    assert sig["confidence"] > noise["confidence"]


def test_astrometric_wobble_feature():
    """Debris (blend wobble + glints) should show a stronger astrometry-
    photometry correlation than a static star with pure jitter noise."""
    records = generate_mock_ztf_dataset(n_clean=25, n_debris=25, seed=17)
    feats = DomainFeatureExtractor().fit_transform(records)
    labels = np.array([int(r["label"]) for r in records])
    corr_debris = feats[labels == 1, 15]
    corr_stars = feats[labels == 0, 15]
    assert np.median(corr_debris) > np.median(corr_stars)


def test_features_degrade_gracefully_without_coords():
    records = generate_mock_ztf_dataset(n_clean=5, n_debris=2, seed=3)
    for r in records:
        r.pop("ra", None), r.pop("dec", None)
    feats = DomainFeatureExtractor().fit_transform(records)
    assert np.all(feats[:, 15] == 0.0) and np.all(feats[:, 16] == 0.0)
    assert np.all(np.isfinite(feats))


def test_baseline_evaluation_runs():
    records = generate_mock_ztf_dataset(n_clean=60, n_debris=15, seed=5)
    results = evaluate_baselines(records, seed=5)
    assert "error" not in results
    assert 0.0 <= results["IsolationForest_ROC_AUC"] <= 1.0
    assert 0.0 <= results["MiniRocket_ROC_AUC"] <= 1.0
    # Supervised baseline should comfortably beat chance on this easy task.
    assert results["MiniRocket_ROC_AUC"] > 0.8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
