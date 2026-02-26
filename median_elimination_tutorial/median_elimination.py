"""
================================================================================
  MEDIAN ELIMINATION ALGORITHM  —  Multi-Armed Bandit (PAC Best-Arm Identification)
================================================================================

Video Reference : https://www.youtube.com/watch?v=3iNaR0Mq1ug

--- PROBLEM SETTING (Multi-Armed Bandit) ---

Imagine you are standing in front of K slot machines (arms). Each arm i has an
unknown reward distribution with mean  μ(i).

Goal :  Find the arm whose mean reward is *close enough* to the best arm's mean,
        using as few total pulls as possible.

Formally, we want to output an arm  â  such that:

        μ(best_arm)  -  μ(â)  ≤  ε          (ε-optimal)

with probability at least  1 - δ.

This is called the **(ε, δ)-PAC** (Probably Approximately Correct) guarantee.

--- NAIVE APPROACH ---

Pull every arm  n = (2/ε²) · ln(2K/δ)  times, then pick the one with the
highest empirical mean.

Sample complexity :  K · n  =  O( K / ε²  ·  ln(K / δ) )

Notice the  ln(K/δ)  factor — the dependence on K *inside* the log is wasteful
because we do not need to be confident about every arm equally. We only care
about not eliminating the best one.

--- MEDIAN ELIMINATION (THE IMPROVEMENT) ---

The key insight: instead of keeping all K arms until the very end, we can
*eliminate the worse half* in each round. This halves the number of arms
every round so the algorithm runs for at most  ⌈log₂ K⌉  rounds.

By carefully splitting the error budget (ε) and confidence budget (δ) across
rounds using geometric series, the total sample complexity becomes:

        O( K / ε²  ·  ln(1 / δ) )

The  ln(K)  factor disappears from the bound — this is *optimal*.

================================================================================
  ALGORITHM  (as described in the video)
================================================================================

INPUTS :   K arms,  ε > 0,  δ > 0

1.  S₁ ← {1, 2, …, K}         (all arms are active)
    ε₁ ← ε / 4
    δ₁ ← δ / 2

2.  WHILE  |Sₗ| > 1 :

    a)  Compute the number of pulls for this round:

              nₗ  =  ⌈ (4 / εₗ²) · ln(3 / δₗ) ⌉

        This comes from Hoeffding's inequality: to guarantee that
        |q̂(a) - μ(a)| ≤ εₗ/2  for a single arm with failure prob δₗ/3,
        we need  n ≥ (1/(εₗ/2)²) · ln(2 / (δₗ/3)) ≈ (4/εₗ²) · ln(3/δₗ).

    b)  Pull every active arm  nₗ  times. Compute empirical mean  q̂ₗ(a).

    c)  Find the MEDIAN  mₗ  of  { q̂ₗ(a)  |  a ∈ Sₗ }.

    d)  Eliminate the bottom half:
              Sₗ₊₁  =  { a ∈ Sₗ  |  q̂ₗ(a) ≥ mₗ }

    e)  Tighten the budgets for the next round:
              εₗ₊₁  =  (3/4) · εₗ
              δₗ₊₁  =  δₗ / 2

    f)  l ← l + 1

3.  OUTPUT the single remaining arm.

--- WHY THE BUDGETS CONVERGE ---

Total ε used  =  Σ εₗ  =  (ε/4) · Σ (3/4)^(l-1)  =  (ε/4) · 1/(1 - 3/4)  =  ε   ✓
Total δ used  =  Σ δₗ  =  (δ/2) · Σ (1/2)^(l-1)  =  (δ/2) · 1/(1 - 1/2)  =  δ   ✓

So the overall guarantee of (ε, δ)-PAC is preserved by construction.

================================================================================
"""

import numpy as np
import math
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAN ELIMINATION CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MedianElimination:
    """
    Median Elimination for (ε, δ)-PAC Best-Arm Identification.

    Parameters
    ----------
    arm_probs : dict
        Mapping from arm_name → true probability (Bernoulli parameter).
        The algorithm doesn't "know" these — it only uses them to generate
        random reward samples (simulating real user clicks / actions).
    epsilon : float
        Desired sub-optimality gap (we accept an arm within ε of the best).
    delta : float
        Maximum allowed failure probability.
    """

    def __init__(self, arm_probs: dict, epsilon: float, delta: float):
        self.arm_probs = arm_probs
        self.epsilon = epsilon
        self.delta = delta

    def run(self, verbose: bool = True):
        """
        Execute the Median Elimination algorithm.

        Returns
        -------
        best_arm : str
            Name of the recommended arm.
        history  : list[dict]
            Per-round diagnostics (active arms, pulls, median, survivors).
        total_pulls : int
            Total number of arm pulls across all rounds.
        """
        S = list(self.arm_probs.keys())     # active arm set
        eps_l = self.epsilon / 4.0          # ε₁ = ε / 4
        del_l = self.delta / 2.0            # δ₁ = δ / 2
        round_num = 1
        total_pulls = 0
        history = []

        if verbose:
            print("=" * 60)
            print("  MEDIAN ELIMINATION ALGORITHM")
            print("=" * 60)
            print(f"  K = {len(S)} arms,  ε = {self.epsilon},  δ = {self.delta}")
            print("=" * 60)

        while len(S) > 1:
            # ── Step (a): Number of pulls per arm in this round ──
            #     nₗ = ⌈ (4 / εₗ²) · ln(3 / δₗ) ⌉
            n_l = int(math.ceil((4.0 / (eps_l ** 2)) * math.log(3.0 / del_l)))

            if verbose:
                print(f"\n── Round {round_num} ──")
                print(f"  Active arms     : {len(S)}")
                print(f"  εₗ = {eps_l:.6f}   δₗ = {del_l:.6f}")
                print(f"  Pulls per arm   : {n_l:,}")

            # ── Step (b): Pull each active arm nₗ times (VECTORISED for speed) ──
            #    np.random.binomial(1, p, size=n_l)  generates n_l Bernoulli
            #    samples in one fast C call — orders of magnitude faster than
            #    a Python for-loop.
            empirical_means = {}
            for arm in S:
                rewards = np.random.binomial(1, self.arm_probs[arm], size=n_l)
                empirical_means[arm] = rewards.mean()

            total_pulls += n_l * len(S)

            # ── Step (c): Find the median of the empirical means ──
            median_val = float(np.median(list(empirical_means.values())))

            # ── Step (d): Keep only arms with empirical mean ≥ median ──
            S_next = [a for a in S if empirical_means[a] >= median_val]

            # Tie-breaking: if more than ceil(|S|/2) survive, keep top half
            max_survivors = math.ceil(len(S) / 2)
            if len(S_next) > max_survivors:
                S_next = sorted(S, key=lambda a: empirical_means[a],
                                reverse=True)[:max_survivors]

            if verbose:
                print(f"  Median reward   : {median_val:.6f}")
                print(f"  Eliminated      : {len(S) - len(S_next)} arms")
                print(f"  Survivors       : {S_next}")

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

            # ── Step (e): Tighten the budgets ──
            S = S_next
            eps_l *= 0.75       # εₗ₊₁ = (3/4) · εₗ
            del_l *= 0.5        # δₗ₊₁ = δₗ / 2
            round_num += 1

        best_arm = S[0]
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  ✅  RECOMMENDED ARM :  {best_arm}")
            print(f"  Total pulls used   :  {total_pulls:,}")
            print(f"{'=' * 60}\n")

        return best_arm, history, total_pulls


# ─────────────────────────────────────────────────────────────────────────────
#  REAL-WORLD SCENARIO :  Online Advertisement Click-Through Rate Optimisation
# ─────────────────────────────────────────────────────────────────────────────
#
#  Imagine you run an e-commerce website and have 10 banner ad designs.
#  Each time an ad is shown to a visitor, the visitor either clicks (reward=1)
#  or does not click (reward=0).  Each ad has an unknown true Click-Through
#  Rate (CTR).
#
#  We want to find the best (or near-best) ad with high confidence while
#  minimising the number of ad impressions wasted on sub-optimal ads.
#

def main():
    np.random.seed(42)

    # True CTRs (unknown to the algorithm — used only to generate samples)
    true_ctrs = {
        "Banner_A": 0.020,   # Basic text ad
        "Banner_B": 0.035,   # Colourful image ad
        "Banner_C": 0.028,   # Animated GIF ad
        "Banner_D": 0.045,   # Video thumbnail ad
        "Banner_E": 0.050,   # Personalised ad  ← BEST
        "Banner_F": 0.015,   # Sidebar ad
        "Banner_G": 0.040,   # Pop-up style ad
        "Banner_H": 0.032,   # Carousel ad
        "Banner_I": 0.022,   # Footer ad
        "Banner_J": 0.038,   # Influencer ad
    }

    print("True CTRs (hidden from the algorithm):")
    print("-" * 40)
    for name, ctr in sorted(true_ctrs.items(), key=lambda x: -x[1]):
        marker = " ◀ BEST" if ctr == max(true_ctrs.values()) else ""
        print(f"  {name} : {ctr:.3f}{marker}")
    print()

    # ── Run Median Elimination ──
    # We want an arm within ε = 0.04 of the best, with 95 % confidence (delta=0.05)
    #
    # ε = 0.04 means we accept any ad whose CTR is at most 4 percentage points
    # worse than the true best. This is a practical trade-off: tighter ε gives
    # a better arm but requires exponentially more samples.
    epsilon = 0.04
    delta   = 0.05

    algo = MedianElimination(true_ctrs, epsilon, delta)
    best_arm, history, total_pulls = algo.run(verbose=True)

    # ── Evaluate the result ──
    best_true = max(true_ctrs.values())
    chosen_true = true_ctrs[best_arm]
    gap = best_true - chosen_true

    print("── EVALUATION ──")
    print(f"  True best CTR        : {best_true:.3f}  (Banner_E)")
    print(f"  Chosen arm CTR       : {chosen_true:.3f}  ({best_arm})")
    print(f"  Sub-optimality gap   : {gap:.4f}")
    print(f"  ε tolerance          : {epsilon}")
    if gap <= epsilon:
        print("  ✅  PAC guarantee satisfied!\n")
    else:
        print("  ❌  Outside tolerance (probability ≤ δ)\n")

    # ── Visualisation ──
    plot_results(history, true_ctrs, best_arm, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(history, true_ctrs, chosen_arm, epsilon):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Arms eliminated per round ---
    rounds = [h["round"] for h in history]
    active = [h["active"] for h in history]
    axes[0].bar(rounds, active, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Active Arms")
    axes[0].set_title("Arms Remaining per Round")
    axes[0].set_xticks(rounds)

    # --- Plot 2: Pulls per arm per round (shows cost growth) ---
    n_ls = [h["n_l"] for h in history]
    axes[1].plot(rounds, n_ls, "o-", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Pulls per arm (nₗ)")
    axes[1].set_title("Sampling Effort per Round")
    axes[1].set_xticks(rounds)
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # --- Plot 3: True CTRs with chosen arm highlighted ---
    names = list(true_ctrs.keys())
    ctrs  = [true_ctrs[n] for n in names]
    colours = ["limegreen" if n == chosen_arm else "salmon" for n in names]
    axes[2].barh(names, ctrs, color=colours, edgecolor="black")
    axes[2].axvline(max(ctrs) - epsilon, linestyle="--", color="grey",
                    label=f"Best − ε ({max(ctrs) - epsilon:.3f})")
    axes[2].set_xlabel("True CTR")
    axes[2].set_title("True CTRs  (green = chosen arm)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("median_elimination_results.png", dpi=150)
    plt.show()
    print("Plot saved to median_elimination_results.png")


if __name__ == "__main__":
    main()
