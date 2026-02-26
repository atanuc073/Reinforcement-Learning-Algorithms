# Median Elimination Algorithm — Complete Tutorial

> **Video Reference**: [Median Elimination – YouTube](https://www.youtube.com/watch?v=3iNaR0Mq1ug)

---

## Table of Contents

1. [The Problem: Best-Arm Identification](#1-the-problem-best-arm-identification)
2. [Naive Approach & Its Limitation](#2-naive-approach--its-limitation)
3. [Median Elimination — The Core Idea](#3-median-elimination--the-core-idea)
4. [Algorithm Step-by-Step](#4-algorithm-step-by-step)
5. [Mathematical Foundations](#5-mathematical-foundations)
   - [Hoeffding's Inequality](#51-hoeffdings-inequality)
   - [Number of Pulls per Round](#52-number-of-pulls-per-round)
   - [Budget Convergence Proof](#53-budget-convergence-proof)
   - [Sample Complexity Analysis](#54-sample-complexity-analysis)
6. [Worked Example (By Hand)](#6-worked-example-by-hand)
7. [Python Implementation](#7-python-implementation)
8. [Running on Real Datasets](#8-running-on-real-datasets)
9. [Results](#9-results)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. The Problem: Best-Arm Identification

Imagine you have **K slot machines** (called **arms**). Each arm `i` gives you a random reward drawn from some distribution with an **unknown** mean `μ(i)`.

**Goal**: Find the arm with the **highest mean reward**, while pulling arms as **few times** as possible.

Since rewards are random, we can never be 100% certain. So we relax the goal:

> **Find an arm `â` such that**:
>
> `μ(best) − μ(â) ≤ ε`
>
> **with probability at least `1 − δ`.**

This is called the **(ε, δ)-PAC guarantee** (Probably Approximately Correct).

- **ε** = how close to the best we need to be (error tolerance)
- **δ** = how much failure probability we allow

---

## 2. Naive Approach & Its Limitation

The simplest approach: pull **every** arm the same number of times `n`, then pick the one with the highest empirical mean.

Using Hoeffding's inequality, we need:

```
n = (2 / ε²) · ln(2K / δ)
```

**Total pulls** = `K · n`  =  **O( K/ε² · ln(K/δ) )**

Notice the **`ln(K)`** factor inside the logarithm — this is wasteful! We're treating every arm equally, but we only care about not accidentally eliminating the *best* arm. The bad arms don't need the same level of statistical confidence.

---

## 3. Median Elimination — The Core Idea

Instead of keeping all K arms until the end, we **eliminate the worse half** in each round:

```
Round 1:  K    arms  →  K/2 survive
Round 2:  K/2  arms  →  K/4 survive
Round 3:  K/4  arms  →  K/8 survive
   ...
Round ⌈log₂K⌉:       →  1 arm left ✓
```

**Key insight**: We split the total error budget `ε` and failure budget `δ` across rounds using **geometric series**, so they converge to exactly `ε` and `δ` at the end.

**Result**: The `ln(K)` factor disappears! The sample complexity becomes:

```
O( K/ε² · ln(1/δ) )    ← OPTIMAL
```

---

## 4. Algorithm Step-by-Step

```
INPUTS:  K arms,  ε > 0,  δ > 0

1. INITIALIZE:
   S₁ ← {1, 2, ..., K}        (all arms active)
   ε₁ ← ε / 4                 (initial error budget)
   δ₁ ← δ / 2                 (initial failure budget)
   l  ← 1                     (round counter)

2. WHILE |Sₗ| > 1:

   (a) Compute pulls per arm:
       nₗ = ⌈ (4 / εₗ²) · ln(3 / δₗ) ⌉

   (b) Pull every active arm nₗ times.
       Compute empirical mean q̂ₗ(a) for each arm a ∈ Sₗ.

   (c) Find the MEDIAN mₗ of all empirical means.

   (d) ELIMINATE the bottom half:
       Sₗ₊₁ = { a ∈ Sₗ  |  q̂ₗ(a) ≥ mₗ }

   (e) TIGHTEN the budgets:
       εₗ₊₁ = (3/4) · εₗ
       δₗ₊₁ = δₗ / 2

   (f) l = l + 1

3. OUTPUT the single remaining arm.
```

---

## 5. Mathematical Foundations

### 5.1 Hoeffding's Inequality

This is the key concentration inequality that powers the algorithm:

> If X₁, X₂, ..., Xₙ are i.i.d. random variables in [0, 1] with mean μ, then:
>
> **P( |X̄ − μ| ≥ t ) ≤ 2 · exp(−2nt²)**

In words: the empirical mean of `n` samples is unlikely to be far from the true mean, and the probability of a large deviation **decreases exponentially** with `n`.

### 5.2 Number of Pulls per Round

We want the empirical mean to be within `εₗ/2` of the true mean for **each** active arm, with failure probability at most `δₗ` for the whole round.

Applying Hoeffding's and a union bound over ≤ 3 "bad events":

```
P( |q̂(a) − μ(a)| ≥ εₗ/2 ) ≤ δₗ/3
```

This gives us:

```
2 · exp(−2n · (εₗ/2)²) ≤ δₗ/3

Solving for n:

n ≥ (1 / (εₗ/2)²) · ln(2 / (δₗ/3))
  = (4 / εₗ²) · ln(6 / δₗ)
  ≈ (4 / εₗ²) · ln(3 / δₗ)      (simplified)
```

Therefore:

```
nₗ = ⌈ (4 / εₗ²) · ln(3 / δₗ) ⌉
```

### 5.3 Budget Convergence Proof

We need to verify that the total error and failure budgets across **all rounds** sum to exactly ε and δ.

**Error budget (ε)**:

```
Σ εₗ  =  ε/4 + ε/4·(3/4) + ε/4·(3/4)² + ...
       =  (ε/4) · Σ (3/4)^(l-1)     for l = 1, 2, ...
       =  (ε/4) · 1/(1 − 3/4)
       =  (ε/4) · 4
       =  ε  ✓
```

**Failure budget (δ)**:

```
Σ δₗ  =  δ/2 + δ/2·(1/2) + δ/2·(1/2)² + ...
       =  (δ/2) · Σ (1/2)^(l-1)     for l = 1, 2, ...
       =  (δ/2) · 1/(1 − 1/2)
       =  (δ/2) · 2
       =  δ  ✓
```

Both infinite geometric series converge to exactly the total budget! This is why the specific constants `ε/4` and `δ/2` are chosen as starting values, and `3/4` and `1/2` as decay factors.

### 5.4 Sample Complexity Analysis

The total number of pulls across all rounds:

```
Total = Σ |Sₗ| · nₗ
```

In round `l`:
- Active arms: `|Sₗ| ≤ K / 2^(l-1)`  (halved each round)
- Pulls per arm: `nₗ = O(1/εₗ² · log(1/δₗ))`

Since εₗ shrinks by `(3/4)` and δₗ shrinks by `(1/2)`, the `1/εₗ²` term grows like `(4/3)^(2l)`, but the number of arms shrinks like `2^(-l)`. The arms shrink **faster** than the pulls grow, so the total is dominated by the first round:

```
Total = O( K/ε² · ln(1/δ) )
```

**Comparison**:

| Method | Sample Complexity |
|--------|------------------|
| Naive (pull all arms equally) | O( K/ε² · **ln(K/δ)** ) |
| **Median Elimination** | O( K/ε² · **ln(1/δ)** ) |

The `ln(K)` factor is completely eliminated! For K = 1000 arms, this is roughly a **3× improvement** in the log factor alone.

---

## 6. Worked Example (By Hand)

Suppose we have **4 arms** with true means:

| Arm | μ |
|-----|------|
| A | 0.30 |
| B | 0.50 |
| C | 0.70 |
| D | 0.90 |

Parameters: `ε = 0.4`, `δ = 0.1`

**Initialization**: `ε₁ = 0.1`, `δ₁ = 0.05`

**Round 1** (4 arms active):
- `n₁ = ⌈(4/0.01) · ln(3/0.05)⌉ = ⌈400 · 4.09⌉ = 1,636 pulls per arm`
- Suppose empirical means: Â=0.31, B̂=0.48, Ĉ=0.72, D̂=0.88
- Median = (0.48 + 0.72) / 2 = 0.60
- Keep arms ≥ 0.60: **C and D survive**, A and B eliminated

**Round 2** (2 arms active):
- `ε₂ = 0.075`, `δ₂ = 0.025`
- `n₂ = ⌈(4/0.005625) · ln(3/0.025)⌉ = ⌈711 · 4.79⌉ = 3,406 pulls per arm`
- Suppose empirical means: Ĉ=0.71, D̂=0.89
- Median = 0.80
- Keep arms ≥ 0.80: **D survives**

**Output**: Arm D (true mean = 0.90) ✅

**Verification**: `μ(D) − μ(D) = 0 ≤ 0.4 = ε` ✓

---

## 7. Python Implementation

### `median_elimination.py` — Simulated Data

This script contains:
- Full mathematical commentary as docstrings
- `MedianElimination` class with vectorized NumPy sampling
- A simulated scenario: **10 banner ads** with different CTRs
- 3-panel matplotlib visualization

**Run it**:
```bash
python median_elimination.py
```

### `median_elimination_datasets.py` — Real CSV Datasets

This script adapts the algorithm for **pre-collected datasets** using bootstrap sampling (sampling with replacement). It runs Median Elimination on:
1. `dataset_ads.csv` — 10 ad creatives (binary click/no-click)
2. `dataset_movies.csv` — 8 movie genres (continuous 0–1 ratings)
3. `dataset_clinical.csv` — 5 drug treatments (binary success/failure)

**Run it**:
```bash
python median_elimination_datasets.py
```

> **Note**: The datasets are in the `ucb_tutorial/` folder. The script references them by absolute path.

---

## 8. Running on Real Datasets

### Dataset 1: Online Ads (10 arms, binary)

Each row = one ad impression. Each column = one ad creative. Value = 1 (clicked) or 0 (not clicked).

| Arm | True CTR |
|-----|----------|
| Popup_Discount | **0.0780** ← Best |
| Social_Carousel | 0.0618 |
| Native_Article | 0.0507 |
| Video_Short | 0.0406 |
| Email_Header | 0.0368 |
| Search_Text | 0.0335 |
| Video_Long | 0.0273 |
| Banner_Red | 0.0224 |
| Banner_Blue | 0.0120 |
| Sidebar_Text | 0.0076 |

### Dataset 2: Movie Genres (8 arms, continuous)

Each row = one user. Each column = average rating for that genre (0 to 1).

| Arm | True Mean Rating |
|-----|-----------------|
| Documentary | **0.7716** ← Best |
| Animation | 0.7226 |
| Drama | 0.6927 |
| Sci-Fi | 0.6450 |
| Comedy | 0.6222 |
| Action | 0.5504 |
| Romance | 0.4723 |
| Horror | 0.3858 |

### Dataset 3: Clinical Trials (5 arms, binary)

Each row = one patient. Each column = treatment outcome (1 = success, 0 = failure).

| Arm | True Success Rate |
|-----|------------------|
| Drug_D (Combo_Therapy) | **0.7280** ← Best |
| Drug_B (New_Compound) | 0.6200 |
| Drug_E (Alternative) | 0.5715 |
| Drug_A (Standard) | 0.4555 |
| Drug_C (Experimental) | 0.3725 |

---

## 9. Results

All three datasets: the algorithm **correctly identified the true best arm**:

| Dataset | ε | Arms | Rounds | Best Arm Found | Gap | Status |
|---------|------|------|--------|----------------|-----|--------|
| Online Ads | 0.01 | 10 | 4 | Popup_Discount | 0.0000 | ✅ |
| Movie Genres | 0.05 | 8 | 3 | Documentary | 0.0000 | ✅ |
| Clinical Trials | 0.05 | 5 | 3 | Drug_D (Combo) | 0.0000 | ✅ |

### Elimination Flow

```
Ads:       10 → 5 → 3 → 2 → 1  (Popup_Discount)
Movies:     8 → 4 → 2 → 1      (Documentary)
Clinical:   5 → 3 → 2 → 1      (Drug_D)
```

---

## 10. Key Takeaways

1. **Median Elimination achieves optimal sample complexity**: `O(K/ε² · ln(1/δ))`, removing the `ln(K)` overhead of the naive approach.

2. **The "halving trick" is the key**: By eliminating half the arms each round, we ensure the algorithm terminates in `⌈log₂ K⌉` rounds.

3. **Geometric budget splitting** is elegant: Starting with `ε₁ = ε/4` (decay `3/4`) and `δ₁ = δ/2` (decay `1/2`) guarantees the series converge to exactly `ε` and `δ`.

4. **Hoeffding's inequality** provides the theoretical backbone — it tells us exactly how many samples we need per round to maintain statistical confidence.

5. **Trade-off**: Smaller ε → more pulls needed (quadratically). Smaller δ → more pulls needed (logarithmically). Choose these parameters based on your domain requirements.

6. **Offline adaptation**: For pre-collected datasets, we use bootstrap sampling (with replacement) to simulate the "pulling" process.

---

## File Structure

```
median_elimination_tutorial/
├── README.md                        ← This file
├── median_elimination.py            ← Algorithm + simulated CTR scenario
└── median_elimination_datasets.py   ← Algorithm on real CSV datasets
```

---

## References

- [Even-Dar, E., Mannor, S., & Mansour, Y. (2006). Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems. *JMLR*.](http://jmlr.org/papers/v7/even-dar06a.html)
- [YouTube: Median Elimination](https://www.youtube.com/watch?v=3iNaR0Mq1ug)
- Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random Variables. *JASA*.
