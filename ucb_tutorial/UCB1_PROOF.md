# UCB1 Regret Bound — Complete Step-by-Step Proof

## The Theorem We Will Prove

> **Theorem (Auer, Cesa-Bianchi, Fischer 2002).**
> Consider a K-armed bandit with rewards in [0, 1]. Let μ₁, μ₂, …, μ_K be the true
> means and μ* = max_i μ_i. Define Δ_i = μ* − μ_i as the suboptimality gap of arm i.
> Then UCB1 satisfies:
>
> **E[R_T] ≤ ∑_{i: Δ_i > 0} (8 ln T)/Δ_i + (1 + π²/3) · ∑_{i: Δ_i > 0} Δ_i**
>
> This is **O(K · log T)** — logarithmic regret.

---

## Roadmap

```
Step 0: Notation & Setup
    ↓
Step 1: Hoeffding's Inequality (the key tool)
    ↓
Step 2: Decompose regret into per-arm counts
    ↓
Step 3: Bound when a suboptimal arm is pulled
    ↓
Step 4: Bound E[N_i(T)] for each suboptimal arm
    ↓
Step 5: Sum up to get total regret bound
    ↓
Step 6: Interpret the result
```

---

## Step 0: Notation & Setup

| Symbol | Meaning |
|--------|---------|
| K | Number of arms |
| T | Total time horizon (number of rounds) |
| μ_i | True (unknown) mean reward of arm i |
| μ* = max_i μ_i | Mean of the best arm |
| i* | Index of the best arm (μ_{i*} = μ*) |
| Δ_i = μ* − μ_i | Suboptimality gap of arm i |
| N_i(t) | Number of times arm i has been pulled through time t |
| X̄_i(t) | Sample mean of arm i after N_i(t) pulls: X̄_i = (1/N_i) ∑ rewards from arm i |
| A(t) | The arm selected at time t |

**UCB1 rule:** At time t, select arm:

```
A(t) = argmax_i [ X̄_i(t-1) + √(2 ln t / N_i(t-1)) ]
```

---

## Step 1: Hoeffding's Inequality

This is the foundational tool. It tells us how far a sample mean can deviate from
the true mean.

### Statement

> Let X₁, X₂, …, X_n be i.i.d. random variables with X_j ∈ [0, 1] and E[X_j] = μ.
> Let X̄_n = (1/n) ∑_{j=1}^{n} X_j. Then for any a > 0:
>
> **P(X̄_n ≥ μ + a) ≤ exp(−2na²)**
>
> **P(X̄_n ≤ μ − a) ≤ exp(−2na²)**

### What this gives us

If we set **a = √(2 ln t / n)**, then:

```
P(X̄_n ≥ μ + √(2 ln t / n)) ≤ exp(−2n · (2 ln t / n))
                              = exp(−4 ln t)
                              = t^{−4}
```

So after n pulls, the probability that the sample mean overshoots the true mean
by more than √(2 ln t / n) is at most **1/t⁴** — extremely small.

Similarly:

```
P(X̄_n ≤ μ − √(2 ln t / n)) ≤ t^{−4}
```

This is why UCB1 uses √(2 ln t / N_i) as the confidence radius — it makes
"bad events" happen with probability at most 1/t⁴.

---

## Step 2: Decompose Regret into Per-Arm Counts

The cumulative regret after T rounds is:

```
R_T = ∑_{t=1}^{T} (μ* − μ_{A(t)})
```

We can rewrite this by grouping by which arm was played:

```
R_T = ∑_{i=1}^{K} Δ_i · N_i(T)
```

**Why?** Each time suboptimal arm i is played, we lose Δ_i = μ* − μ_i in expected
reward. If arm i is played N_i(T) times total, the total loss from arm i is Δ_i · N_i(T).

Taking expectations:

```
┌────────────────────────────────────────────────┐
│                                                │
│   E[R_T] = ∑_{i: Δ_i > 0}  Δ_i · E[N_i(T)]   │
│                                                │
└────────────────────────────────────────────────┘
```

**Key insight:** To bound the total regret, we just need to bound **E[N_i(T)]** — the
expected number of times each suboptimal arm i is pulled.

---

## Step 3: When Is a Suboptimal Arm Pulled?

A suboptimal arm i (with Δ_i > 0) is pulled at time t only if its UCB is the highest:

```
UCB_i(t) ≥ UCB_{i*}(t)
```

Expanding:

```
X̄_i + √(2 ln t / N_i) ≥ X̄_{i*} + √(2 ln t / N_{i*})
```

Define the confidence radii:

```
c_{i,s}(t) = √(2 ln t / s)     — radius when arm i has been pulled s times
```

For arm i to be pulled, at least one of three things must be true. We label
them as events:

---

### Event E₁: The best arm's sample mean is too low

```
E₁: X̄_{i*} ≤ μ* − c_{i*, N_{i*}}(t)
```

The best arm's sample mean has fallen **below** its lower confidence bound.
By Hoeffding: **P(E₁) ≤ t^{−4}**

---

### Event E₂: The suboptimal arm's sample mean is too high

```
E₂: X̄_i ≥ μ_i + c_{i, N_i}(t)
```

Arm i's sample mean has risen **above** its upper confidence bound.
By Hoeffding: **P(E₂) ≤ t^{−4}**

---

### Event E₃: Arm i hasn't been pulled enough

```
E₃: N_i(t) < ℓ    where ℓ = ⌈(8 ln T) / Δ_i²⌉
```

If arm i has been pulled fewer than ℓ times, its confidence interval is still
wide enough that UCB_i could exceed UCB_{i*} even when E₁ and E₂ don't occur.

---

### Why these three events cover everything

**Claim:** If E₁, E₂, and E₃ all fail (i.e., none of them occur), then arm i
**cannot** be pulled.

**Proof of claim:**

Assume ¬E₁, ¬E₂, ¬E₃ all hold. Then:

1. ¬E₁ means: X̄_{i*} > μ* − c_{i*, N_{i*}}(t)
   → **UCB_{i*} = X̄_{i*} + c_{i*, N_{i*}}(t) > μ***
   (actually, UCB_{i*} ≥ X̄_{i*} + c > μ* − c + c = μ*)

2. ¬E₂ means: X̄_i < μ_i + c_{i, N_i}(t)
   → **UCB_i = X̄_i + c_{i, N_i}(t) < μ_i + 2·c_{i, N_i}(t)**

3. ¬E₃ means: N_i ≥ ℓ = ⌈(8 ln T) / Δ_i²⌉
   → c_{i, N_i}(t) = √(2 ln t / N_i) ≤ √(2 ln T / ℓ)

   Now substitute ℓ = 8 ln T / Δ_i²:
   ```
   c_{i, N_i}(t) ≤ √(2 ln T / (8 ln T / Δ_i²))
                   = √(2 · Δ_i² / 8)
                   = √(Δ_i² / 4)
                   = Δ_i / 2
   ```

Combining 2 and 3:
```
UCB_i < μ_i + 2 · (Δ_i / 2) = μ_i + Δ_i = μ*
```

But from 1: UCB_{i*} ≥ μ*

Therefore: **UCB_{i*} ≥ μ* > UCB_i**, so the algorithm picks i* over i. ∎

---

## Step 4: Bound E[N_i(T)]

We now count how many times suboptimal arm i can be pulled. We decompose:

```
N_i(T) = ∑_{t=1}^{T} 𝟙{A(t) = i}
```

From Step 3, at each time t, arm i is pulled only if E₁ ∨ E₂ ∨ E₃ occurs.

### Part A: Contribution from E₃ (not enough pulls)

Arm i can be pulled at most **ℓ = ⌈(8 ln T) / Δ_i²⌉** times due to this event
(after ℓ pulls, E₃ no longer holds).

This contributes at most:

```
⌈(8 ln T) / Δ_i²⌉
```

### Part B: Contribution from E₁ or E₂ (bad concentration events)

Even after arm i has been pulled ℓ times, it could still be pulled if E₁ or E₂
occurs. We bound the expected number of such occurrences.

At time t, using union bound over all possible values of N_i and N_{i*}:

```
P(arm i pulled at time t, and N_i ≥ ℓ)
    ≤ P(E₁ at time t) + P(E₂ at time t)
    ≤ ∑_{s=1}^{t} P(X̄_{i*,s} ≤ μ* − √(2 ln t / s))
      + ∑_{s=ℓ}^{t} P(X̄_{i,s} ≥ μ_i + √(2 ln t / s))
```

But we can bound this more directly. For each time t and each possible
count s (number of pulls of arm i) and s' (number of pulls of best arm):

```
P(E₁ or E₂) ≤ ∑_{s=1}^{t-1} ∑_{s'=1}^{t-1} [P(X̄_{i*,s'} ≤ μ* − c_{s'}(t)) 
                                                  + P(X̄_{i,s} ≥ μ_i + c_s(t))]
```

By Hoeffding (from Step 1), each probability is at most t^{−4}, so:

```
P(E₁ or E₂) ≤ ∑_{s=1}^{t-1} ∑_{s'=1}^{t-1} 2·t^{−4}
             ≤ t² · 2·t^{−4}
             = 2·t^{−2}
```

### Summing over all time steps

```
Total contribution from E₁, E₂ = ∑_{t=1}^{∞} 2·t^{−2}
                                = 2 · π²/6
                                = π²/3
```

(We use the Basel series: ∑_{t=1}^{∞} 1/t² = π²/6.)

### Combining Parts A and B

```
┌───────────────────────────────────────────────────────┐
│                                                       │
│   E[N_i(T)] ≤ ⌈(8 ln T) / Δ_i²⌉ + π²/3             │
│                                                       │
│   Simplifying the ceiling:                            │
│   E[N_i(T)] ≤ (8 ln T) / Δ_i² + 1 + π²/3            │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## Step 5: Compute Total Regret

Plugging back into the regret decomposition from Step 2:

```
E[R_T] = ∑_{i: Δ_i > 0} Δ_i · E[N_i(T)]

       ≤ ∑_{i: Δ_i > 0} Δ_i · [(8 ln T) / Δ_i² + 1 + π²/3]

       = ∑_{i: Δ_i > 0} [(8 ln T) / Δ_i + Δ_i · (1 + π²/3)]
```

Separating the two sums:

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│                        K                        K            │
│   E[R_T]  ≤   8 ln T · ∑  (1/Δ_i)  +  (1+π²/3) ∑  Δ_i     │
│                       i=1                      i=1           │
│                      i≠i*                     i≠i*           │
│                                                              │
│           =  O(K · log T)                                    │
│                                                              │
│                                                              │
│   This is the UCB1 Theorem. ∎                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 6: Understanding the Result

### Term by term

| Term | What it means |
|------|--------------|
| 8 ln T / Δ_i | The "necessary exploration" cost. Arms close to optimal (small Δ_i) are hard to distinguish → more pulls needed. |
| (1 + π²/3) · Δ_i | A small constant cost from the rare "bad events" (concentration failures). Bounded by a constant. |

### Growth rate

```
   Regret
     ▲
     │
 800 ┤                                          ╱ Linear O(T) — bad algorithms
     │                                       ╱
 600 ┤                                    ╱
     │                                 ╱
 400 ┤                              ╱
     │                           ╱
 200 ┤        ............................  Logarithmic O(log T) — UCB1
     │      .·
 100 ┤    .·
     │  .·
     │.·
     └───────┬──────┬──────┬──────┬─────► T
          2000   4000   6000   8000  10000
```

- **Log T grows MUCH slower than T**. After 10,000 rounds, log(10000) ≈ 9.2.
- This means the **average regret per round → 0** as T → ∞.
- UCB1 is **consistent**: it eventually figures out the best arm.

### Is this the best possible?

Yes (up to constants)! The **Lai-Robbins lower bound (1985)** proves:

> For any algorithm, E[R_T] ≥ ∑_i (Δ_i / KL(μ_i, μ*)) · ln T

where KL(·,·) is the KL divergence. Since KL(μ_i, μ*) ≤ 2/Δ_i² for bounded
rewards, this gives a **Ω(log T)** lower bound.

UCB1's O(log T) **matches this lower bound** in order — it's *rate-optimal*.

---

## Summary of the Proof Flow

```
Hoeffding's Inequality
    │
    ▼
"Bad events" (E₁, E₂) happen with probability ≤ t^{−4}
    │
    ▼
After ℓ = O(ln T / Δ_i²) pulls, arm i's UCB < μ* (when no bad events)
    │
    ▼
E[N_i(T)] ≤ 8 ln T / Δ_i²  +  constant
    │
    ▼
E[R_T] = ∑ Δ_i · E[N_i(T)] ≤ 8 ln T · ∑ 1/Δ_i  +  constant
    │
    ▼
Regret = O(K · log T)    ✓
```

---

## Conditions Under Which UCB1 Works

The proof above relies on specific assumptions. Here is exactly where each one
is used, and what happens when it breaks.

### Assumption 1: Bounded Rewards — X_t ∈ [0, 1]

- **Used in:** Step 1 (Hoeffding's Inequality requires bounded random variables)
- **If violated:** The concentration bound exp(−2na²) no longer holds. With
  unbounded or heavy-tailed rewards, the sample mean converges much slower.
- ❌ **Breaks for:** Gaussian with unknown variance, Pareto, log-normal rewards
- ✅ **Fix:** If rewards are in [a, b], rescale via r' = (r − a)/(b − a). For
  truly unbounded rewards, use **Robust UCB** or **Median-of-Means UCB**.

### Assumption 2: Stationarity — μ_i is constant over time

- **Used in:** Step 2 (regret decomposition assumes μ* is fixed) and Step 4
  (sample mean X̄_i converges to a fixed μ_i)
- **If violated:** The sample mean converges to a time-average, not the current
  mean. The best arm might change, making past data misleading.
- ❌ **Breaks for:** Trending rewards, seasonal patterns, adversarial settings
- ✅ **Fix:** **Sliding Window UCB** (only use recent data), **Discounted UCB**
  (exponentially weight recent observations), or **EXP3** (adversarial setting).

### Assumption 3: Independence — rewards are i.i.d. within each arm

- **Used in:** Step 1 (Hoeffding requires X₁, X₂, …, Xₙ to be independent)
- **If violated:** The concentration rate changes. Positively correlated samples
  make the sample mean converge slower than 1/√n.
- ❌ **Breaks for:** Time-series data, correlated user sessions, sequential trials
  with carryover effects
- ✅ **Fix:** Use modified concentration inequalities for dependent data
  (e.g., martingale-based bounds), or **restless bandits** formulations.

### Assumption 4: No Context — the optimal arm is the same in every round

- **Used in:** Step 2 (single μ* for all rounds)
- **If violated:** The best arm depends on a feature vector x_t (e.g., user
  demographics). A single global ranking of arms is meaningless.
- ❌ **Breaks for:** Personalized recommendations, user-specific ad targeting
- ✅ **Fix:** **LinUCB** (linear contextual bandits), **Kernel UCB**, or
  **Contextual Thompson Sampling**.

### Assumption 5: Finite, Fixed Arm Set — K arms known in advance

- **Used in:** Step 0 (initialization requires playing each arm once) and Step 4
  (sum over i = 1 to K)
- **If violated:** If K is very large or infinite, initialization alone costs too
  much. If arms appear/disappear, the indexing scheme breaks.
- ❌ **Breaks for:** Continuous action spaces, dynamically changing action sets
- ✅ **Fix:** **GP-UCB** (Gaussian Process UCB) for continuous arms,
  **Combinatorial bandits** for exponentially large arm sets.

### Assumption 6: Bandit Feedback — you observe reward of the chosen arm only

- **Used in:** Algorithm definition (update rule only sees one reward per round)
- This is NOT a restrictive assumption — it's the *definition* of the bandit
  setting. UCB1 does NOT need to see what other arms would have given.
- ✅ **If you observe all arms' rewards:** You're in the **full information**
  setting, which is strictly easier. UCB1 still works but is overkill.

### Quick Reference Table

| Assumption | Where Used in Proof | What Breaks | Use Instead |
|-----------|-------|-------|------------|
| Bounded [0,1] rewards | Hoeffding (Step 1) | Concentration bound | Robust UCB, Median-of-Means |
| Stationary means | Regret decomp (Step 2) | Sample mean misleading | Sliding Window UCB, EXP3 |
| i.i.d. rewards | Hoeffding (Step 1) | Convergence rate wrong | Martingale bounds |
| No context/state | Single μ* (Step 2) | Best arm varies | LinUCB, Contextual bandits |
| Finite fixed K arms | Initialization (Step 0) | Can't try all arms | GP-UCB, Continuum bandits |
| Bandit feedback | Update rule | N/A (not restrictive) | Still works with more info |

---

## Reference

Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time Analysis of the
Multiarmed Bandit Problem.* Machine Learning, 47(2-3), 235–256.
