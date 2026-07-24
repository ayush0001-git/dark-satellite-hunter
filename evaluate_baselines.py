"""
Baseline Evaluation (`evaluate_baselines.py`)

Benchmarks the unsupervised Isolation Forest (domain features) and the
supervised MiniRocket+Ridge classifier on a stratified held-out split of
clean vs debris-injected light curves.

Supervised metrics require ground-truth labels, so real (unlabeled) ZTF data
can only be benchmarked after synthetic injection - use mock mode for that.
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data.mock_generator import generate_mock_ztf_dataset
from src.models.baselines import evaluate_baselines


def main():
    parser = argparse.ArgumentParser(description="Dark Satellite Hunter | Baseline Benchmarking")
    parser.add_argument("--use_mock", type=str, default="True")
    parser.add_argument("--n_clean", type=int, default=500)
    parser.add_argument("--n_debris", type=int, default=120)
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_mock.lower() not in ("true", "1", "yes"):
        print("[!] Baseline benchmarking needs labeled data. Real ZTF light curves are "
            "unlabeled; run with --use_mock True (labels come from synthetic injection).")
        sys.exit(1)

    print("\n[*] Evaluating baselines for Space Domain Awareness...")
    print(f"  * {args.n_clean} clean targets (RR Lyrae, EBs, QSOs, constants)")
    print(f"  * {args.n_debris} tumbling-debris injections (diffuse + specular glints)")
    records = generate_mock_ztf_dataset(
        n_clean=args.n_clean, n_debris=args.n_debris, seed=args.seed
    )

    results = evaluate_baselines(records, test_size=args.test_size, seed=args.seed)
    if "error" in results:
        print(f"[!] {results['error']}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("[+] BASELINE RESULTS (stratified held-out split: "
          f"{results['n_train']} train / {results['n_test']} test)")
    print("=" * 65)
    print(f"  * [Unsupervised] Isolation Forest ROC-AUC       : {results['IsolationForest_ROC_AUC']:.4f}")
    print(f"  * [Unsupervised] Isolation Forest PR-AUC        : {results['IsolationForest_PR_AUC']:.4f}")
    print(f"  * [Unsupervised] Isolation Forest Recall@1%FPR  : {results['IsolationForest_Recall@1%FPR']*100:.2f}%")
    print("-" * 65)
    print(f"  * [Supervised]   MiniRocket + Ridge ROC-AUC     : {results['MiniRocket_ROC_AUC']:.4f}")
    print(f"  * [Supervised]   MiniRocket + Ridge PR-AUC      : {results['MiniRocket_PR_AUC']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
