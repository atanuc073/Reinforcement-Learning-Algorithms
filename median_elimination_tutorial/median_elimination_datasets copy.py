"""
================================================================================
  MEDIAN ELIMINATION ON REAL DATASETS
================================================================================

This script applies the Median Elimination algorithm to three real-world
bandit datasets:

  1. dataset_ads.csv      — 10 ad creatives, binary click/no-click
  2. dataset_movies.csv   — 8 movie genres, continuous 0–1 ratings
  3. dataset_clinical.csv — 5 drug treatments, binary success/failure

ADAPTATION FOR FIXED DATASETS:
  The original Median Elimination generates fresh random samples. Here the
  data is pre-collected, so "pulling an arm n times" = sampling n rows
  (with replacement) from that arm's column.  This is standard practice
  in offline bandit evaluation.

================================================================================
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAN ELIMINATION FOR DATASETS
# ─────────────────────────────────────────────────────────────────────────────

class MedianEliminationDataset:
    """
    Median Elimination algorithm adapted for pre-collected datasets.

    Instead of generating rewards on-the-fly, each "pull" draws a random
    row from the arm's column (sampling with replacement), which is the
    standard bootstrap approach for offline bandit evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        Each column = one arm, each row = one observed reward.
    epsilon : float
        Desired sub-optimality gap.
    delta : float
        Maximum failure probability.
    """

    def __init__(self, data: pd.DataFrame, epsilon: float, delta: float):
        self.data = data
        self.epsilon = epsilon
        self.delta = delta

    def run(self, verbose: bool = True):
        arm_names = list(self.data.columns)
        S = list(arm_names)
        eps_l = self.epsilon / 4.0
        del_l = self.delta / 2.0
        round_num = 1
        total_pulls = 0
        history = []

        if verbose:
            print("=" * 65)
            print("  MEDIAN ELIMINATION  (Dataset Mode)")
            print("=" * 65)
            print(f"  Arms        : {len(S)}")
            print(f"  Data rows   : {len(self.data)}")
            print(f"  ε = {self.epsilon},  δ = {self.delta}")
            print("=" * 65)

        while len(S) > 1:
            # nₗ = ⌈ (4 / εₗ²) · ln(3 / δₗ) ⌉
            n_l = int(math.ceil((4.0 / (eps_l ** 2)) * math.log(3.0 / del_l)))

            # Cap n_l at dataset size to avoid excessive memory use
            # (sampling with replacement, but no need to exceed dataset)
            n_l = min(n_l, len(self.data))

            if verbose:
                print(f"\n── Round {round_num} ──")
                print(f"  Active arms   : {len(S)}  {S}")
                print(f"  εₗ = {eps_l:.6f}   δₗ = {del_l:.6f}")
                print(f"  Pulls per arm : {n_l:,}")

            # Sample n_l rows (with replacement) from each active arm
            empirical_means = {}
            for arm in S:
                samples = self.data[arm].sample(n=n_l, replace=True,
                                                random_state=round_num).values
                empirical_means[arm] = float(samples.mean())

            total_pulls += n_l * len(S)

            # Find median and eliminate bottom half
            median_val = float(np.median(list(empirical_means.values())))
            S_next = [a for a in S if empirical_means[a] >= median_val]

            max_survivors = math.ceil(len(S) / 2)
            if len(S_next) > max_survivors:
                S_next = sorted(S, key=lambda a: empirical_means[a],
                                reverse=True)[:max_survivors]

            if verbose:
                print(f"  Empirical means:")
                for arm in S:
                    marker = " ✓" if arm in S_next else " ✗"
                    print(f"    {arm:30s} : {empirical_means[arm]:.4f}{marker}")
                print(f"  Median        : {median_val:.4f}")
                print(f"  Eliminated    : {len(S) - len(S_next)} arms")

            history.append({
                "round": round_num,
                "active": len(S),
                "n_l": n_l,
                "eps_l": eps_l,
                "del_l": del_l,
                "median": median_val,
                "means": dict(empirical_means),
                "survivors": list(S_next),
            })

            S = S_next
            eps_l *= 0.75
            del_l *= 0.5
            round_num += 1

        best_arm = S[0]
        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  ✅  RECOMMENDED ARM :  {best_arm}")
            print(f"  Total samples used :  {total_pulls:,}")
            print(f"{'=' * 65}\n")

        return best_arm, history, total_pulls


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_results(title, data, best_arm, history, save_path):
    """Create a 2-panel plot: true column means + elimination rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Panel 1: True column means (bar chart)
    col_means = data.mean().sort_values(ascending=True)
    colours = ["limegreen" if c == best_arm else "steelblue" for c in col_means.index]
    axes[0].barh(col_means.index, col_means.values, color=colours, edgecolor="black")
    axes[0].set_xlabel("Mean Reward (from full dataset)")
    axes[0].set_title("Arm Means  (green = chosen)")

    # Panel 2: Arms remaining per round
    rounds = [h["round"] for h in history]
    active = [h["active"] for h in history]
    axes[1].bar(rounds, active, color="darkorange", edgecolor="black")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Active Arms")
    axes[1].set_title("Elimination Progress")
    axes[1].set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}\n")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — RUN ON ALL THREE DATASETS
# ─────────────────────────────────────────────────────────────────────────────

def run_on_dataset(csv_path, label, epsilon, delta):
    """Load a CSV, run Median Elimination, plot, and return the result."""
    print(f"\n{'#' * 70}")
    print(f"#  DATASET: {label}")
    print(f"#  File   : {csv_path}")
    print(f"{'#' * 70}")

    df = pd.read_csv(csv_path)

    # Drop the ID / timestep column (first column)
    id_col = df.columns[0]
    df = df.drop(columns=[id_col])

    print(f"\nTrue arm means (computed from entire dataset):")
    print("-" * 50)
    for col in df.columns:
        print(f"  {col:30s} : {df[col].mean():.4f}")

    # Determine true best arm
    true_best = df.mean().idxmax()
    print(f"\n  → True best arm: {true_best}  (mean = {df[true_best].mean():.4f})")
    print()

    algo = MedianEliminationDataset(df, epsilon, delta)
    best_arm, history, total_pulls = algo.run(verbose=True)

    # Evaluate
    gap = df[true_best].mean() - df[best_arm].mean()
    print("── EVALUATION ──")
    print(f"  True best arm    : {true_best}  (mean = {df[true_best].mean():.4f})")
    print(f"  Chosen arm       : {best_arm}  (mean = {df[best_arm].mean():.4f})")
    print(f"  Gap              : {gap:.4f}")
    print(f"  ε tolerance      : {epsilon}")
    if gap <= epsilon:
        print("  ✅  Within ε tolerance!")
    else:
        print("  ⚠️  Outside ε tolerance (probability ≤ δ)")

    # Save plot
    base_dir = os.path.dirname(csv_path)
    plot_name = os.path.join(base_dir, f"median_elim_{label.lower().replace(' ', '_')}.png")
    plot_dataset_results(f"Median Elimination — {label}", df, best_arm, history, plot_name)

    return best_arm


def main():
    np.random.seed(42)

    base = r"D:\MY_WORK\Reinforcement Learning tutorial\ucb_tutorial"

    # ── Dataset 1: Online Ads (10 arms, binary) ──
    run_on_dataset(
        csv_path=os.path.join(base, "dataset_ads.csv"),
        label="Online Ads",
        epsilon=0.01,     # ads have low CTRs, so tight ε
        delta=0.05,
    )

    # ── Dataset 2: Movie Recommendations (8 arms, continuous 0–1) ──
    run_on_dataset(
        csv_path=os.path.join(base, "dataset_movies.csv"),
        label="Movie Genres",
        epsilon=0.05,     # ratings are 0–1, modest gap
        delta=0.05,
    )

    # ── Dataset 3: Clinical Trials (5 arms, binary) ──
    run_on_dataset(
        csv_path=os.path.join(base, "dataset_clinical.csv"),
        label="Clinical Trials",
        epsilon=0.05,
        delta=0.05,
    )


if __name__ == "__main__":
    main()
