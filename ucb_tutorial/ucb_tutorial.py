"""
╔════════════════════════════════════════════════════════════════╗
║     UCB1 (Upper Confidence Bound) — Complete Tutorial         ║
║     From Scratch: Multi-Armed Bandits & the UCB1 Theorem      ║
╚════════════════════════════════════════════════════════════════╝

This script implements and visualizes:
  1. The Multi-Armed Bandit problem
  2. The UCB1 algorithm
  3. Comparison with Epsilon-Greedy and Random strategies
  4. Regret analysis and confidence interval visualization

Author: UCB1 Reinforcement Learning Tutorial
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves plots without blocking
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os

# ═══════════════════════════════════════════════════════════════
# SECTION 1: THE BANDIT ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class BanditArm:
    """
    A single arm of the multi-armed bandit.
    
    Each arm has a TRUE (hidden) probability of giving reward = 1.
    The agent doesn't know this probability — it must learn by pulling.
    
    Reward model: Bernoulli distribution
        - With probability p: reward = 1
        - With probability (1-p): reward = 0
    """
    
    def __init__(self, true_mean: float, name: str = ""):
        """
        Args:
            true_mean: The true (hidden) probability of reward.
                       Must be in [0, 1] for Bernoulli rewards.
            name: Optional descriptive name for this arm.
        """
        assert 0 <= true_mean <= 1, f"True mean must be in [0,1], got {true_mean}"
        self.true_mean = true_mean
        self.name = name
    
    def pull(self) -> float:
        """Pull this arm and receive a stochastic reward."""
        return 1.0 if np.random.random() < self.true_mean else 0.0
    
    def __repr__(self):
        return f"Arm('{self.name}', μ={self.true_mean:.2f})"


class MultiarmedBandit:
    """
    The Multi-Armed Bandit environment.
    
    Contains K arms, each with a different (unknown) reward probability.
    The agent interacts by selecting arms and receiving rewards.
    """
    
    def __init__(self, arms: List[BanditArm]):
        self.arms = arms
        self.K = len(arms)
        self.best_arm = max(range(self.K), key=lambda i: arms[i].true_mean)
        self.best_mean = arms[self.best_arm].true_mean
        
        print(f"\n{'='*60}")
        print(f"  Multi-Armed Bandit Created with {self.K} arms")
        print(f"{'='*60}")
        for i, arm in enumerate(arms):
            marker = " ⭐ BEST" if i == self.best_arm else ""
            print(f"  Arm {i}: μ = {arm.true_mean:.3f}  {arm.name}{marker}")
        print(f"{'='*60}\n")
    
    def pull(self, arm_index: int) -> float:
        """Pull arm at given index and return reward."""
        return self.arms[arm_index].pull()
    
    def suboptimality_gap(self, arm_index: int) -> float:
        """Δᵢ = μ* - μᵢ  (how much worse arm i is vs the best)"""
        return self.best_mean - self.arms[arm_index].true_mean


# ═══════════════════════════════════════════════════════════════
# SECTION 2: THE UCB1 ALGORITHM
# ═══════════════════════════════════════════════════════════════

class UCB1Agent:
    """
    UCB1 (Upper Confidence Bound) Algorithm
    
    The UCB1 formula for arm i at time t:
    
        UCBᵢ(t) = X̄ᵢ + √(2·ln(t) / Nᵢ)
                  ────   ─────────────────
                 exploit    exploration
                  term        bonus
    
    Where:
        X̄ᵢ  = sample mean of rewards from arm i
        Nᵢ  = number of times arm i has been pulled
        t   = current time step
    
    Key insight: "Optimism in the face of uncertainty"
        - Arms we know little about get a LARGE bonus → explore them
        - Arms we know a lot about are judged on merit → exploit them
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)       # Nᵢ: times each arm pulled
        self.sum_rewards = np.zeros(n_arms)              # Total reward from each arm
        self.total_steps = 0                             # t: total time steps
        
        # For visualization/logging
        self.history_arms = []
        self.history_rewards = []
        self.history_ucb_values = []
    
    def get_sample_means(self) -> np.ndarray:
        """X̄ᵢ = (sum of rewards from arm i) / (times arm i was pulled)"""
        means = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                means[i] = self.sum_rewards[i] / self.counts[i]
        return means
    
    def get_exploration_bonus(self) -> np.ndarray:
        """
        The exploration term: √(2·ln(t) / Nᵢ)
        
        This is derived from Hoeffding's Inequality:
            P(|X̄ₙ - μ| ≥ u) ≤ 2·exp(-2nu²)
        
        Setting u = √(2·ln(t)/Nᵢ) makes the probability ≤ 2/t⁴
        which is small enough for the regret proof to work.
        """
        bonus = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.counts[i] > 0:
                bonus[i] = np.sqrt(2 * np.log(self.total_steps) / self.counts[i])
            else:
                bonus[i] = float('inf')  # Never-pulled arms have infinite bonus
        return bonus
    
    def get_ucb_values(self) -> np.ndarray:
        """UCBᵢ(t) = X̄ᵢ + √(2·ln(t) / Nᵢ)"""
        return self.get_sample_means() + self.get_exploration_bonus()
    
    def select_arm(self) -> int:
        """
        Select the arm with the highest UCB value.
        
        Phase 1 (Initialization): Pull each arm once (round-robin)
        Phase 2 (Main loop): Pick argmax UCBᵢ(t)
        """
        # Phase 1: Initialization — pull each arm at least once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        # Phase 2: Select arm with highest UCB
        ucb_values = self.get_ucb_values()
        self.history_ucb_values.append(ucb_values.copy())
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, reward: float):
        """Update statistics after pulling an arm."""
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_steps += 1
        self.history_arms.append(arm)
        self.history_rewards.append(reward)
    
    def step(self, bandit: MultiarmedBandit) -> Tuple[int, float]:
        """Complete one step: select arm, pull it, update stats."""
        arm = self.select_arm()
        reward = bandit.pull(arm)
        self.update(arm, reward)
        return arm, reward


# ═══════════════════════════════════════════════════════════════
# SECTION 3: COMPARISON STRATEGIES
# ═══════════════════════════════════════════════════════════════

class EpsilonGreedyAgent:
    """
    Epsilon-Greedy Strategy (for comparison)
    
    - With probability ε: pick a RANDOM arm (explore)
    - With probability 1-ε: pick the arm with highest sample mean (exploit)
    
    Problem: ε is a hyperparameter that must be tuned.
    Too high → too much exploration, too low → might miss the best arm.
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.sum_rewards = np.zeros(n_arms)
        self.total_steps = 0
        self.history_arms = []
        self.history_rewards = []
    
    def select_arm(self) -> int:
        # Always try each arm at least once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # Explore
        else:
            means = self.sum_rewards / np.maximum(self.counts, 1)
            return int(np.argmax(means))  # Exploit
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_steps += 1
        self.history_arms.append(arm)
        self.history_rewards.append(reward)
    
    def step(self, bandit: MultiarmedBandit) -> Tuple[int, float]:
        arm = self.select_arm()
        reward = bandit.pull(arm)
        self.update(arm, reward)
        return arm, reward


class RandomAgent:
    """
    Random Strategy (baseline)
    
    Always picks a random arm. No learning at all.
    Expected regret: O(T) — linear, the worst possible.
    """
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.sum_rewards = np.zeros(n_arms)
        self.total_steps = 0
        self.history_arms = []
        self.history_rewards = []
    
    def select_arm(self) -> int:
        return np.random.randint(self.n_arms)
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_steps += 1
        self.history_arms.append(arm)
        self.history_rewards.append(reward)
    
    def step(self, bandit: MultiarmedBandit) -> Tuple[int, float]:
        arm = self.select_arm()
        reward = bandit.pull(arm)
        self.update(arm, reward)
        return arm, reward


# ═══════════════════════════════════════════════════════════════
# SECTION 4: RUNNING EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def compute_cumulative_regret(agent, bandit: MultiarmedBandit) -> np.ndarray:
    """
    Compute the cumulative regret of an agent's history.
    
    Regret at each step = μ* - μ_{arm chosen}
    Cumulative regret = running sum of per-step regrets
    """
    regrets = []
    for arm in agent.history_arms:
        regrets.append(bandit.suboptimality_gap(arm))
    return np.cumsum(regrets)


def run_experiment(bandit: MultiarmedBandit, agent, n_steps: int, 
                   label: str = "", verbose: bool = False):
    """Run an agent on a bandit for n_steps."""
    for t in range(n_steps):
        arm, reward = agent.step(bandit)
        
        if verbose and t < 20:
            if hasattr(agent, 'get_ucb_values') and agent.total_steps > bandit.K:
                ucb_vals = agent.get_ucb_values()
                means = agent.get_sample_means()
                bonus = agent.get_exploration_bonus()
                print(f"  t={t+1:3d} | Selected arm {arm} | Reward={reward:.0f} | "
                      f"Means={np.array2string(means, precision=3)} | "
                      f"Bonus={np.array2string(bonus, precision=3)} | "
                      f"UCB={np.array2string(ucb_vals, precision=3)}")
            else:
                print(f"  t={t+1:3d} | Selected arm {arm} | Reward={reward:.0f}")
    
    cum_regret = compute_cumulative_regret(agent, bandit)
    
    if label:
        print(f"\n  {label} Results after {n_steps} steps:")
        print(f"    Total reward: {sum(agent.history_rewards):.0f}")
        print(f"    Cumulative regret: {cum_regret[-1]:.2f}")
        print(f"    Arm pulls: {agent.counts}")
        if hasattr(agent, 'get_sample_means'):
            print(f"    Sample means: {np.array2string(agent.get_sample_means(), precision=3)}")
        print(f"    True means:  {[f'{a.true_mean:.3f}' for a in bandit.arms]}")
    
    return cum_regret


# ═══════════════════════════════════════════════════════════════
# SECTION 5: VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_regret_comparison(bandit, n_steps=10000, n_runs=20):
    """
    EXPERIMENT: Compare cumulative regret of UCB1 vs Epsilon-Greedy vs Random.
    
    This is the key result:
    - UCB1 regret grows LOGARITHMICALLY (slow)
    - ε-Greedy regret grows LINEARLY (fast)
    - Random regret grows LINEARLY (fastest)
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT: Regret Comparison")
    print(f"  Running {n_runs} trials of {n_steps} steps each...")
    print("=" * 60)
    
    all_regrets = {'UCB1': [], 'ε-Greedy (ε=0.1)': [], 'Random': []}
    
    for run in range(n_runs):
        # UCB1
        agent_ucb = UCB1Agent(bandit.K)
        regret_ucb = run_experiment(bandit, agent_ucb, n_steps)
        all_regrets['UCB1'].append(regret_ucb)
        
        # Epsilon Greedy
        agent_eg = EpsilonGreedyAgent(bandit.K, epsilon=0.1)
        regret_eg = run_experiment(bandit, agent_eg, n_steps)
        all_regrets['ε-Greedy (ε=0.1)'].append(regret_eg)
        
        # Random
        agent_rand = RandomAgent(bandit.K)
        regret_rand = run_experiment(bandit, agent_rand, n_steps)
        all_regrets['Random'].append(regret_rand)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'UCB1': '#2ecc71', 'ε-Greedy (ε=0.1)': '#e74c3c', 'Random': '#95a5a6'}
    
    # Left plot: Cumulative regret
    ax = axes[0]
    for label, regrets in all_regrets.items():
        mean_regret = np.mean(regrets, axis=0)
        std_regret = np.std(regrets, axis=0)
        steps = np.arange(1, n_steps + 1)
        ax.plot(steps, mean_regret, label=label, color=colors[label], linewidth=2)
        ax.fill_between(steps, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret,
                        alpha=0.15, color=colors[label])
    
    # Add theoretical O(log t) curve for reference
    theoretical = 8 * np.log(steps) * sum(1/bandit.suboptimality_gap(i) 
                                           for i in range(bandit.K) 
                                           if bandit.suboptimality_gap(i) > 0)
    ax.plot(steps, theoretical, '--', color='#2ecc71', alpha=0.5, linewidth=1,
            label='Theoretical O(log t) bound')
    
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('Cumulative Regret', fontsize=12)
    ax.set_title('Cumulative Regret Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right plot: Average regret per step (should → 0 for UCB1)
    ax = axes[1]
    for label, regrets in all_regrets.items():
        mean_regret = np.mean(regrets, axis=0)
        avg_per_step = mean_regret / np.arange(1, n_steps + 1)
        ax.plot(steps, avg_per_step, label=label, color=colors[label], linewidth=2)
    
    ax.set_xlabel('Time Step (t)', fontsize=12)
    ax.set_ylabel('Average Regret per Step', fontsize=12)
    ax.set_title('Average Regret → 0 for UCB1 (Consistency)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "regret_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved plot: {save_path}")
    plt.show()


def plot_ucb_detailed(bandit, n_steps=5000):
    """
    EXPERIMENT: Visualize UCB1 internals — sample means, confidence bounds,
    and arm selection over time.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT: UCB1 Detailed Visualization")
    print(f"  Running UCB1 for {n_steps} steps...")
    print("=" * 60)
    
    agent = UCB1Agent(bandit.K)
    
    # Track UCB components over time
    means_history = []
    lower_bounds = []
    upper_bounds = []
    selected_arms = []
    
    for t in range(n_steps):
        arm, reward = agent.step(bandit)
        selected_arms.append(arm)
        
        means = agent.get_sample_means()
        bonus = agent.get_exploration_bonus()
        
        means_history.append(means.copy())
        upper_bounds.append((means + bonus).copy())
        lower_bounds.append((means - bonus).copy())
    
    means_history = np.array(means_history)
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    colors = plt.cm.Set2(np.linspace(0, 1, bandit.K))
    steps = np.arange(1, n_steps + 1)
    
    # Plot 1: Sample means with confidence intervals
    ax = axes[0]
    for i in range(bandit.K):
        label = f"Arm {i} (μ={bandit.arms[i].true_mean:.2f})"
        ax.plot(steps, means_history[:, i], color=colors[i], label=label, linewidth=1.5)
        ax.fill_between(steps, lower_bounds[:, i], upper_bounds[:, i],
                        color=colors[i], alpha=0.1)
        # True mean as dashed line
        ax.axhline(y=bandit.arms[i].true_mean, color=colors[i], 
                   linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('UCB1: Sample Means with Confidence Intervals\n'
                 '(Dashed = true means, Shaded = confidence bounds)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Exploration bonus over time
    ax = axes[1]
    for i in range(bandit.K):
        bonus = upper_bounds[:, i] - means_history[:, i]
        ax.plot(steps, bonus, color=colors[i], 
                label=f"Arm {i} bonus", linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Exploration Bonus √(2ln(t)/Nᵢ)', fontsize=11)
    ax.set_title('Exploration Bonus Shrinks Over Time\n'
                 '(Less-played arms keep higher bonus)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Arm selection frequency (rolling window)
    ax = axes[2]
    window = 50
    for i in range(bandit.K):
        selections = np.array([1 if a == i else 0 for a in selected_arms])
        if len(selections) >= window:
            rolling_freq = np.convolve(selections, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window, n_steps + 1), rolling_freq,
                    color=colors[i], 
                    label=f"Arm {i} (μ={bandit.arms[i].true_mean:.2f})",
                    linewidth=1.5)
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel(f'Selection Frequency (rolling {window})', fontsize=11)
    ax.set_title('Arm Selection Frequency Over Time\n'
                 '(UCB1 converges to the best arm)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "ucb1_detailed.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved plot: {save_path}")
    plt.show()


def plot_hoeffding_demo():
    """
    DEMONSTRATION: Hoeffding's Inequality in action.
    
    Shows how the confidence interval shrinks as we collect more samples.
    This is the mathematical foundation of UCB1.
    """
    print("\n" + "=" * 60)
    print("  DEMO: Hoeffding's Inequality Visualization")
    print("=" * 60)
    
    np.random.seed(42)
    true_mean = 0.7  # True probability
    max_samples = 500
    
    # Collect samples
    samples = np.random.binomial(1, true_mean, max_samples)
    
    # Compute running statistics
    sample_means = np.cumsum(samples) / np.arange(1, max_samples + 1)
    ns = np.arange(1, max_samples + 1)
    
    # Hoeffding confidence interval (95% → δ = 0.05)
    delta = 0.05
    hoeffding_bound = np.sqrt(np.log(2 / delta) / (2 * ns))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Sample mean convergence with confidence band
    ax = axes[0]
    ax.plot(ns, sample_means, color='#3498db', linewidth=1.5, label='Sample Mean X̄ₙ')
    ax.fill_between(ns, 
                    sample_means - hoeffding_bound,
                    sample_means + hoeffding_bound,
                    alpha=0.2, color='#3498db', label='95% Hoeffding CI')
    ax.axhline(y=true_mean, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'True Mean μ = {true_mean}')
    
    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title("Hoeffding's Inequality: Confidence Interval Shrinks\n"
                 "as More Samples Are Collected", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right: Width of confidence interval
    ax = axes[1]
    ax.plot(ns, 2 * hoeffding_bound, color='#e74c3c', linewidth=2, 
            label='CI Width = 2√(ln(2/δ)/(2n))')
    
    # Mark specific points
    for n_mark in [10, 50, 100, 200, 500]:
        idx = n_mark - 1
        width = 2 * hoeffding_bound[idx]
        ax.plot(n_mark, width, 'o', color='#2c3e50', markersize=8)
        ax.annotate(f'n={n_mark}\nwidth={width:.3f}', 
                    xy=(n_mark, width), xytext=(n_mark + 30, width + 0.05),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Confidence Interval Width', fontsize=12)
    ax.set_title('Confidence Interval Width Decreases\n'
                 'Proportional to 1/√n', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "hoeffding_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved plot: {save_path}")
    plt.show()


def plot_arm_selection_pie(bandit, n_steps=5000):
    """Show how UCB1 allocates pulls across arms."""
    agent = UCB1Agent(bandit.K)
    run_experiment(bandit, agent, n_steps)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = [f"Arm {i}\nμ={bandit.arms[i].true_mean:.2f}\n({agent.counts[i]} pulls)" 
              for i in range(bandit.K)]
    colors = plt.cm.Set2(np.linspace(0, 1, bandit.K))
    
    wedges, texts, autotexts = ax.pie(
        agent.counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}
    )
    
    ax.set_title(f'UCB1 Arm Selection Distribution\n'
                 f'(After {n_steps} steps)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "arm_selection_pie.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved plot: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# SECTION 6: STEP-BY-STEP WALKTHROUGH
# ═══════════════════════════════════════════════════════════════

def step_by_step_walkthrough():
    """
    Walk through the first 20 steps of UCB1 to build intuition.
    
    This shows exactly what happens inside the algorithm:
    - How it initializes (plays each arm once)
    - How UCB values are computed
    - Why it switches between arms
    """
    print("\n" + "=" * 60)
    print("  STEP-BY-STEP UCB1 WALKTHROUGH")
    print("=" * 60)
    
    # Simple 3-arm bandit
    arms = [
        BanditArm(0.3, "Low"),     # Bad arm
        BanditArm(0.5, "Medium"),  # Decent arm
        BanditArm(0.8, "High"),    # Best arm
    ]
    bandit = MultiarmedBandit(arms)
    agent = UCB1Agent(bandit.K)
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    print(f"\n{'Step':>4} | {'Action':>8} | {'Reward':>6} | {'Counts':>12} | "
          f"{'Sample Means':>20} | {'Explore Bonus':>20} | {'UCB Values':>20}")
    print("-" * 110)
    
    for t in range(20):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        
        means = agent.get_sample_means()
        
        if agent.total_steps > bandit.K:
            bonus = agent.get_exploration_bonus()
            ucb = agent.get_ucb_values()
            bonus_str = np.array2string(bonus, precision=3, suppress_small=True)
            ucb_str = np.array2string(ucb, precision=3, suppress_small=True)
        else:
            bonus_str = "  (initializing)  "
            ucb_str = "  (initializing)  "
        
        means_str = np.array2string(means, precision=3, suppress_small=True)
        counts_str = str(agent.counts)
        
        phase = "INIT" if t < bandit.K else "UCB"
        print(f"  {t+1:3d} | {phase} → {arm:1d} | {reward:5.1f} | {counts_str:>12} | "
              f"{means_str:>20} | {bonus_str:>20} | {ucb_str:>20}")
    
    print(f"\n  After 20 steps:")
    print(f"    Best arm (true): Arm {bandit.best_arm} (μ={bandit.best_mean:.2f})")
    print(f"    Most pulled arm: Arm {np.argmax(agent.counts)} "
          f"({agent.counts[np.argmax(agent.counts)]} times)")
    print(f"    ✅ UCB1 is already focusing on the best arm!")


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN — RUN ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def main():
    """
    Run the complete UCB1 tutorial with all experiments.
    """
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   🎰  UCB1 Upper Confidence Bound — Complete Tutorial  🎰  ║
    ║                                                            ║
    ║   Topics covered:                                          ║
    ║   1. Step-by-step walkthrough of UCB1                      ║
    ║   2. Hoeffding's Inequality visualization                  ║
    ║   3. UCB1 detailed internals                               ║
    ║   4. Regret comparison: UCB1 vs ε-Greedy vs Random         ║
    ║   5. Arm selection distribution                            ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ── Part 1: Step-by-step walkthrough ──
    step_by_step_walkthrough()
    
    # ── Part 2: Hoeffding's Inequality demo ──
    plot_hoeffding_demo()
    
    # ── Create the bandit for remaining experiments ──
    arms = [
        BanditArm(0.1, "Very Bad"),
        BanditArm(0.3, "Bad"),
        BanditArm(0.5, "Medium"),
        BanditArm(0.7, "Good"),
        BanditArm(0.9, "Excellent"),
    ]
    bandit = MultiarmedBandit(arms)
    
    # ── Part 3: UCB1 detailed visualization ──
    plot_ucb_detailed(bandit, n_steps=1000)
    
    # ── Part 4: Regret comparison ──
    plot_regret_comparison(bandit, n_steps=10000, n_runs=20)
    
    # ── Part 5: Arm selection distribution ──
    plot_arm_selection_pie(bandit, n_steps=5000)
    
    # ── Summary ──
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║                     TUTORIAL COMPLETE!                      ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║   Key Takeaways:                                           ║
    ║                                                            ║
    ║   1. UCB1 = Sample Mean + Exploration Bonus                ║
    ║                                                            ║
    ║   2. Exploration Bonus = √(2·ln(t) / Nᵢ)                  ║
    ║      → Derived from Hoeffding's Inequality                 ║
    ║      → Shrinks as arm is played more                       ║
    ║                                                            ║
    ║   3. UCB1 achieves O(log T) regret                         ║
    ║      → Near-optimal! (lower bound is Ω(log T))            ║
    ║      → Much better than ε-Greedy or Random                 ║
    ║                                                            ║
    ║   4. No hyperparameters needed!                             ║
    ║      → Unlike ε-Greedy which needs tuning                  ║
    ║                                                            ║
    ║   5. "Optimism in the Face of Uncertainty"                  ║
    ║      → Unknown arms get the benefit of the doubt           ║
    ║      → This naturally balances explore vs exploit           ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
