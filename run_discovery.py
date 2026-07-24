"""
End-to-End Discovery & Cross-Match Pipeline (`run_discovery.py`)

Stages (no ground-truth labels are used at any stage of discovery):

1. Load light curves (mock survey or real ZTF Parquet).
2. Build object representations:
     - with `--ckpt_path`: PatchTST-MAE [CLS] latents + masked-patch
       reconstruction errors (batched inference), or
     - without: hand-crafted domain features, with the reduced chi^2
       variability statistic as the reconstruction-error analogue.
3. Positional cross-match against the reference star/QSO catalog (and TLEs
   if provided) to establish the "known object" reference population.
4. Calibrate the hybrid anomaly scorer on that reference population.
5. Score everything; tag unmatched high-surprisal objects as
   UNCORRELATED_ANOMALY; estimate tumbling periods and astrometric rates.
6. Export `candidates_final.csv` + JSON summary. If simulation ground truth
   is available (mock mode), a validation ROC-AUC is reported separately -
   it never feeds back into the scores.
"""
import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.mock_generator import build_reference_catalog, generate_mock_ztf_dataset
from src.models.baselines import DomainFeatureExtractor, infer_glint_material
from src.pipeline.anomaly_scorer import HybridAnomalyScorer
from src.pipeline.crossmatch import CrossmatchEngine
from src.pipeline.orbital_fit import estimate_glint_periodicity, fit_preliminary_orbit

SEQ_LEN = 256
MIN_REFERENCE_OBJECTS = 30


def str2bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def load_survey(args):
    """Returns (records, reference_catalog)."""
    if str2bool(args.use_mock) or not args.parquet:
        print(f"  * Simulating survey field: {args.n_clean} cataloged sources, "
              f"{args.n_debris} uncataloged debris blends (seed={args.seed})")
        records = generate_mock_ztf_dataset(
            n_clean=args.n_clean, n_debris=args.n_debris, seed=args.seed
        )
        catalog = build_reference_catalog(records)
        return records, catalog

    from src.data.ztf_dataset import PolarsZTFDataset
    print(f"  * Loading Parquet survey data from {args.parquet!r} (limit={args.limit})")
    records = PolarsZTFDataset(args.parquet, seq_len=SEQ_LEN).load_records(limit=args.limit)

    catalog = []
    if args.catalog_csv and os.path.exists(args.catalog_csv):
        cat_df = pd.read_csv(args.catalog_csv)
        catalog = [
            {"catalog_id": str(r["catalog_id"]), "ra_deg": float(r["ra_deg"]),
             "dec_deg": float(r["dec_deg"]), "obj_class": str(r.get("obj_class", "Star"))}
            for _, r in cat_df.iterrows()
        ]
        print(f"  * Reference catalog: {len(catalog)} sources from {args.catalog_csv!r}")
    else:
        print("  [!] No reference catalog supplied (--catalog_csv). "
              "Positional star matching disabled; scorer will calibrate on the full sample.")
    return records, catalog


def build_representations(records, args):
    """
    Returns a dict of representation components:
        feats        [N, 14]  domain features (always computed)
        recon_feat   [N]      log10 reduced chi^2 (variability surprisal)
        mae_latents  [N, D]   PatchTST-MAE [CLS] latents (checkpoint mode only)
        mae_recon    [N]      masked-patch reconstruction errors (checkpoint only)
        mode         str
    """
    print("  * Computing domain-feature representations "
          "(14 astrophysical variability statistics per object).")
    feats = DomainFeatureExtractor().fit_transform(records).astype(np.float64)
    reps = {"feats": feats, "recon_feat": feats[:, 10], "mode": "domain_features"}

    if not args.ckpt_path:
        return reps
    if not os.path.exists(args.ckpt_path):
        print(f"  [!] Checkpoint not found at {args.ckpt_path!r} - "
              "continuing with domain features only.")
        return reps

    import torch
    from torch.utils.data import DataLoader
    from src.data.ztf_dataset import LightCurveDataset, collate_lightcurves
    from src.models.patchtst_mae import PatchTSTMaskedAutoencoder

    print(f"  * PatchTST-MAE inference from checkpoint {args.ckpt_path!r} on {args.device}")
    try:
        model = PatchTSTMaskedAutoencoder.load_from_checkpoint(args.ckpt_path, map_location=args.device)
    except Exception:
        model = PatchTSTMaskedAutoencoder()
        state = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict(state.get("state_dict", state))
    model = model.to(args.device).eval()

    ds = LightCurveDataset(records, seq_len=SEQ_LEN, train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_lightcurves)

    latents, recon = [], []
    torch.manual_seed(args.seed)  # reproducible random masking in recon error
    with torch.no_grad():
        for batch in tqdm(loader, desc="MAE inference", unit="batch"):
            x = batch["x"].to(args.device)
            t = batch["t"].to(args.device)
            obs_mask = batch["obs_mask"].to(args.device)
            latents.append(model.extract_cls_latent(x, t).cpu().numpy())
            recon.append(model.per_sample_recon_error(x, t, obs_mask).cpu().numpy())

    reps["mae_latents"] = np.vstack(latents).astype(np.float64)
    reps["mae_recon"] = np.concatenate(recon).astype(np.float64)
    reps["mode"] = "ensemble_features+mae"
    return reps


def main():
    parser = argparse.ArgumentParser(description="Dark Satellite Hunter | Discovery Pipeline")
    parser.add_argument("--use_mock", type=str, default="True")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Real light-curve archive: .parquet/.csv file or a directory of them")
    parser.add_argument("--offline_mode", type=str, default="True",
                        help="False = live-verify high-surprisal candidates against "
                             "Simbad and the Minor Planet Center (SkyBoT) via astroquery")
    parser.add_argument("--limit", type=int, default=5000, help="Max objects from Parquet")
    parser.add_argument("--catalog_csv", type=str, default=None,
                        help="Reference catalog CSV (catalog_id, ra_deg, dec_deg[, obj_class])")
    parser.add_argument("--tle_path", type=str, default=None, help="TLE file for satellite matching")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Trained PatchTST-MAE checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_clean", type=int, default=800)
    parser.add_argument("--n_debris", type=int, default=150)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--anomaly_threshold", type=float, default=2.0,
                        help="Surprisal (robust z units) above which unmatched objects are flagged")
    parser.add_argument("--mae_weight", type=float, default=0.5,
                        help="Weight of the MAE surprisal component in checkpoint mode "
                             "(0 = features only, 1 = MAE only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_csv", type=str, default="candidates_final.csv")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("[*] DARK SATELLITE HUNTER | ANOMALY DISCOVERY & CROSS-MATCH")
    print("=" * 65)

    # 1. Data + catalog ------------------------------------------------------
    records, catalog = load_survey(args)
    N = len(records)
    print(f"  * Light curves loaded: {N} | Reference catalog size: {len(catalog)}")
    if N == 0:
        raise RuntimeError("No light curves to process.")

    # 2. Representations -----------------------------------------------------
    reps = build_representations(records, args)
    rep_mode = reps["mode"]

    # 3. Positional reference selection (no labels involved) ------------------
    offline = str2bool(args.offline_mode)
    if not offline:
        print("  * LIVE API MODE: high-surprisal candidates will be verified online "
              "against Simbad and the Minor Planet Center (SkyBoT).")
    if args.tle_path:
        print(f"  * TLE catalog loaded for SGP4 satellite matching: {args.tle_path!r}")
    engine = CrossmatchEngine(
        reference_catalog=catalog, tle_path=args.tle_path,
        offline_mode=offline, anomaly_threshold=args.anomaly_threshold,
    )
    if args.tle_path:
        print(f"    -> {len(engine.tle_list)} TLE element sets parsed.")

    mean_ra = np.array([float(np.mean(r["ra"])) if "ra" in r else np.nan for r in records])
    mean_dec = np.array([float(np.mean(r["dec"])) if "dec" in r else np.nan for r in records])
    has_coords = ~np.isnan(mean_ra)

    matched_ref = np.zeros(N, dtype=bool)
    if catalog and has_coords.any():
        for i in tqdm(np.flatnonzero(has_coords), desc="Reference cross-match", unit="obj"):
            hit, _, _ = engine.check_reference_catalog(mean_ra[i], mean_dec[i])
            matched_ref[i] = hit

    n_matched = int(matched_ref.sum())
    if n_matched >= MIN_REFERENCE_OBJECTS:
        ref_idx = np.flatnonzero(matched_ref)
        print(f"  * Scorer reference population: {n_matched} catalog-matched objects.")
    else:
        ref_idx = np.arange(N)
        print(f"  [!] Only {n_matched} catalog matches (<{MIN_REFERENCE_OBJECTS}); "
              "calibrating on the full sample with robust statistics instead.")

    # 4. Calibrate + score ----------------------------------------------------
    # Feature-based surprisal (always available).
    scorer_feat = HybridAnomalyScorer(alpha=0.5, beta=0.5, density_method="mahalanobis")
    scorer_feat.fit(reps["feats"][ref_idx], reps["recon_feat"][ref_idx])
    s_feat = scorer_feat.score(reps["recon_feat"], reps["feats"])["surprisal_score"]

    # MAE-based surprisal (checkpoint mode) - combined as a weighted ensemble.
    s_mae = None
    if "mae_latents" in reps:
        scorer_mae = HybridAnomalyScorer(alpha=0.5, beta=0.5, density_method="mahalanobis")
        scorer_mae.fit(reps["mae_latents"][ref_idx], reps["mae_recon"][ref_idx])
        s_mae = scorer_mae.score(reps["mae_recon"], reps["mae_latents"])["surprisal_score"]
        w = float(np.clip(args.mae_weight, 0.0, 1.0))
        surprisal = (1.0 - w) * s_feat + w * s_mae
        print(f"  * Ensemble scoring: {1-w:.2f} x domain-features + {w:.2f} x MAE surprisal")
    else:
        surprisal = s_feat

    # 5. Tag + kinematics ------------------------------------------------------
    print("  * Tagging candidates and estimating kinematics...")
    rows = []
    for i in tqdm(range(N), desc="Vetting candidates", unit="obj"):
        rec = records[i]
        ra_i = float(mean_ra[i]) if has_coords[i] else float("nan")
        dec_i = float(mean_dec[i]) if has_coords[i] else float("nan")
        mjd_mid = float(np.median(rec["raw_t"])) if "raw_t" in rec else 59100.0

        period_info = estimate_glint_periodicity(rec["t"], rec["mag"])
        if has_coords[i]:
            kin = fit_preliminary_orbit(rec["t"], rec["ra"], rec["dec"])
        else:
            kin = fit_preliminary_orbit(np.empty(0), np.empty(0), np.empty(0))

        if has_coords[i]:
            tag = engine.tag_candidate(
                object_id=rec["object_id"], ra_deg=ra_i, dec_deg=dec_i,
                mjd_obs=mjd_mid, surprisal_score=float(surprisal[i]),
            )
        else:
            tag = {"status": "NO_ASTROMETRY", "matched_catalog": "NONE",
                   "confidence": "LOW", "match_separation_arcsec": None}

        rows.append({
            "object_id": rec["object_id"],
            "RA": round(ra_i, 5) if has_coords[i] else None,
            "Dec": round(dec_i, 5) if has_coords[i] else None,
            "Surprisal_Score": round(float(surprisal[i]), 3),
            "Status": tag["status"],
            "Matched_Catalog": tag["matched_catalog"],
            "Match_Sep_arcsec": tag.get("match_separation_arcsec"),
            "Confidence": tag["confidence"],
            "Period_Guess_Hours": round(period_info["period_hours"], 2),
            "Phase_Sharpness": round(period_info["phase_sharpness"], 3),
            "Period_False_Alarm_Prob": round(period_info["false_alarm_prob"], 4),
            "Num_Specular_Glints": int(period_info["num_glints"]),
            "Material_Inference": infer_glint_material(
                float(reps["feats"][i, 14]), int(period_info["num_glints"])
            ),
            "Orbital_Regime": kin["regime"],
            "Rate_deg_day": round(kin["angular_rate_deg_day"], 5),
            "Rate_arcsec_sec": round(kin["angular_rate_arcsec_sec"], 5),
        })

    df = pd.DataFrame(rows).sort_values("Surprisal_Score", ascending=False)
    top_df = df.head(args.top_k)
    top_df.to_csv(args.output_csv, index=False)

    # 6. Summary + honest validation -------------------------------------------
    uncorrelated = df[df["Status"] == "UNCORRELATED_ANOMALY"]
    summary = {
        "total_scanned": int(N),
        "representation_mode": rep_mode,
        "reference_population": "catalog_matched" if n_matched >= MIN_REFERENCE_OBJECTS else "full_sample_robust",
        "n_reference": int(len(ref_idx)),
        "anomaly_threshold": args.anomaly_threshold,
        "top_k_exported": int(len(top_df)),
        "status_breakdown": df["Status"].value_counts().to_dict(),
        "total_uncorrelated_anomalies": int(len(uncorrelated)),
        "top_uncorrelated_anomalies": uncorrelated.head(10)[
            ["object_id", "Surprisal_Score", "Period_Guess_Hours", "Phase_Sharpness", "Orbital_Regime"]
        ].to_dict(orient="records"),
    }

    labels = np.array([int(r.get("label", -1)) for r in records])
    if set(np.unique(labels)) >= {0, 1}:
        from sklearn.metrics import average_precision_score, roc_auc_score
        val_auc = roc_auc_score(labels, surprisal)
        val_pr = average_precision_score(labels, surprisal)
        component_aucs = {"domain_features": round(float(roc_auc_score(labels, s_feat)), 4)}
        if s_mae is not None:
            component_aucs["mae_latents"] = round(float(roc_auc_score(labels, s_mae)), 4)
        # Detection metrics for the discrete flag decision:
        flagged = df.set_index("object_id")["Status"].eq("UNCORRELATED_ANOMALY")
        flag_by_obj = np.array([bool(flagged.get(r["object_id"], False)) for r in records])
        precision = float(labels[flag_by_obj].mean()) if flag_by_obj.any() else 0.0
        recall = float(flag_by_obj[labels == 1].mean()) if (labels == 1).any() else 0.0
        summary["validation_against_simulation_truth"] = {
            "note": "Ground-truth labels used ONLY here, after scoring - never during discovery.",
            "surprisal_ROC_AUC": round(float(val_auc), 4),
            "surprisal_PR_AUC": round(float(val_pr), 4),
            "component_ROC_AUC": component_aucs,
            "flag_precision": round(precision, 4),
            "flag_recall": round(recall, 4),
        }

    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("[+] DISCOVERY PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  * Representation mode        : {rep_mode}")
    print(f"  * Top candidates exported to : {args.output_csv}")
    print(f"  * Summary saved to           : {summary_path}")
    print(f"  * Uncorrelated anomalies     : {len(uncorrelated)}")
    if "validation_against_simulation_truth" in summary:
        v = summary["validation_against_simulation_truth"]
        print(f"  * VALIDATION (labels never used in scoring):")
        print(f"      Surprisal ROC-AUC {v['surprisal_ROC_AUC']:.4f} | PR-AUC {v['surprisal_PR_AUC']:.4f}")
        print(f"      Flag precision {v['flag_precision']:.2%} | recall {v['flag_recall']:.2%}")
    print("\nTop 5 candidates:")
    print(top_df[["object_id", "Surprisal_Score", "Status", "Period_Guess_Hours",
                  "Phase_Sharpness", "Orbital_Regime"]].head(5).to_string(index=False))
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
