"""
╔════════════════════════════════════════════════════════════════╗
║   UCB1 on Real-World Inspired Datasets                        ║
║   Run UCB1 on practical scenarios with real data patterns      ║
╚════════════════════════════════════════════════════════════════╝

Three dataset scenarios:
  1. Online Ad Click-Through Rate Optimization (Bernoulli rewards)
  2. Movie Recommendation (real ratings from a generated MovieLens-like dataset)
  3. Clinical Trial Drug Selection (Gaussian rewards)
  4. Load your OWN dataset from CSV

Usage:
    python ucb_with_datasets.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
from typing import List, Tuple

# ═══════════════════════════════════════════════════════════════
# REUSABLE UCB1 AGENT (from main tutorial)
# ═══════════════════════════════════════════════════════════════

class UCB1Agent:
    """UCB1 algorithm — see README.md for full explanation."""
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.sum_rewards = np.zeros(n_arms)
        self.total_steps = 0
        self.history_arms = []
        self.history_rewards = []
    
    def select_arm(self) -> int:
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        
        means = self.sum_rewards / self.counts
        bonus = np.sqrt(2 * np.log(self.total_steps) / self.counts)
        ucb_values = means + bonus
        return int(np.argmax(ucb_values))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_steps += 1
        self.history_arms.append(arm)
        self.history_rewards.append(reward)
    
    def get_sample_means(self):
        return np.where(self.counts > 0, self.sum_rewards / self.counts, 0)


class EpsilonGreedyAgent:
    """ε-Greedy for comparison."""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.sum_rewards = np.zeros(n_arms)
        self.total_steps = 0
        self.history_arms = []
        self.history_rewards = []
    
    def select_arm(self) -> int:
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        means = self.sum_rewards / np.maximum(self.counts, 1)
        return int(np.argmax(means))
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sum_rewards[arm] += reward
        self.total_steps += 1
        self.history_arms.append(arm)
        self.history_rewards.append(reward)
    
    def get_sample_means(self):
        return np.where(self.counts > 0, self.sum_rewards / self.counts, 0)


# ═══════════════════════════════════════════════════════════════
# DATASET 1: ONLINE AD CLICK-THROUGH RATE OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

def generate_ad_dataset(n_samples=10000, seed=42):
    """
    Simulates an online advertising scenario.
    
    You have 10 different ad creatives. Each ad has a different 
    click-through rate (CTR). Your goal: find the best ad to 
    maximize clicks (revenue).
    
    This generates a CSV file with pre-computed rewards for each 
    ad at each time step, so you can replay the same dataset with
    different algorithms.
    """
    np.random.seed(seed)
    
    # 10 ad creatives with different CTRs (realistic values: 0.5% to 8%)
    ads = {
        "Banner_Blue":       0.015,   # 1.5% CTR
        "Banner_Red":        0.022,   # 2.2% CTR
        "Video_Short":       0.045,   # 4.5% CTR
        "Video_Long":        0.032,   # 3.2% CTR
        "Popup_Discount":    0.078,   # 7.8% CTR — BEST
        "Sidebar_Text":      0.008,   # 0.8% CTR
        "Native_Article":    0.055,   # 5.5% CTR
        "Email_Header":      0.041,   # 4.1% CTR
        "Social_Carousel":   0.062,   # 6.2% CTR
        "Search_Text":       0.035,   # 3.5% CTR
    }
    
    ad_names = list(ads.keys())
    ctrs = list(ads.values())
    
    # Generate reward matrix: each row = one time step, each col = one ad
    # reward[t][i] = 1 if showing ad i at time t would get a click, else 0
    rewards = np.zeros((n_samples, len(ads)))
    for i, ctr in enumerate(ctrs):
        rewards[:, i] = np.random.binomial(1, ctr, n_samples).astype(float)
    
    # Save to CSV
    filepath = os.path.join(os.path.dirname(__file__), "dataset_ads.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestep"] + ad_names)
        for t in range(n_samples):
            writer.writerow([t] + list(rewards[t]))
    
    print(f"  📁 Generated: {filepath}")
    print(f"     {n_samples} time steps × {len(ads)} ads")
    print(f"     True CTRs: {dict(zip(ad_names, [f'{c:.1%}' for c in ctrs]))}")
    
    return ad_names, ctrs, rewards, filepath


# ═══════════════════════════════════════════════════════════════
# DATASET 2: MOVIE RECOMMENDATION
# ═══════════════════════════════════════════════════════════════

def generate_movie_dataset(n_users=5000, seed=42):
    """
    Simulates a movie recommendation scenario (MovieLens-inspired).
    
    You have 8 movie genres to recommend. Each genre has a 
    different average rating from users. Rewards are normalized
    ratings (0 to 1).
    """
    np.random.seed(seed)
    
    genres = {
        "Action":       (3.2, 0.8),   # (mean_rating, std) out of 5
        "Comedy":       (3.5, 0.7),
        "Drama":        (3.8, 0.9),
        "Sci-Fi":       (3.6, 1.0),
        "Romance":      (2.9, 0.6),
        "Horror":       (2.5, 1.2),
        "Documentary":  (4.1, 0.5),   # BEST
        "Animation":    (3.9, 0.6),
    }
    
    genre_names = list(genres.keys())
    
    # Generate ratings (clipped to [1, 5], then normalized to [0, 1])
    rewards = np.zeros((n_users, len(genres)))
    for i, (name, (mean, std)) in enumerate(genres.items()):
        raw_ratings = np.random.normal(mean, std, n_users)
        clipped = np.clip(raw_ratings, 1.0, 5.0)
        rewards[:, i] = (clipped - 1.0) / 4.0  # Normalize to [0, 1]
    
    true_means = [np.mean(rewards[:, i]) for i in range(len(genres))]
    
    # Save to CSV
    filepath = os.path.join(os.path.dirname(__file__), "dataset_movies.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id"] + genre_names)
        for t in range(n_users):
            writer.writerow([t] + [f"{r:.4f}" for r in rewards[t]])
    
    print(f"  📁 Generated: {filepath}")
    print(f"     {n_users} users × {len(genres)} genres")
    print(f"     True mean ratings (normalized): "
          f"{dict(zip(genre_names, [f'{m:.3f}' for m in true_means]))}")
    
    return genre_names, true_means, rewards, filepath


# ═══════════════════════════════════════════════════════════════
# DATASET 3: CLINICAL TRIAL DRUG SELECTION
# ═══════════════════════════════════════════════════════════════

def generate_clinical_dataset(n_patients=2000, seed=42):
    """
    Simulates a clinical trial where you must choose between
    5 drugs for treating a condition. Each drug has a different
    efficacy (probability of successful treatment).
    
    This is a HIGH-STAKES scenario where exploration-exploitation
    tradeoff matters a lot — you want to find the best drug fast
    but can't waste too many patients on bad drugs.
    """
    np.random.seed(seed)
    
    drugs = {
        "Drug_A (Standard)":     0.45,   # 45% success rate
        "Drug_B (New_Compound)": 0.62,   # 62% — promising!
        "Drug_C (Experimental)": 0.38,   # 38% — not great
        "Drug_D (Combo_Therapy)":0.71,   # 71% — BEST
        "Drug_E (Alternative)":  0.55,   # 55% — decent
    }
    
    drug_names = list(drugs.keys())
    efficacies = list(drugs.values())
    
    rewards = np.zeros((n_patients, len(drugs)))
    for i, eff in enumerate(efficacies):
        rewards[:, i] = np.random.binomial(1, eff, n_patients).astype(float)
    
    filepath = os.path.join(os.path.dirname(__file__), "dataset_clinical.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id"] + drug_names)
        for t in range(n_patients):
            writer.writerow([t] + list(rewards[t]))
    
    print(f"  📁 Generated: {filepath}")
    print(f"     {n_patients} patients × {len(drugs)} drugs")
    print(f"     True efficacies: {dict(zip(drug_names, [f'{e:.0%}' for e in efficacies]))}")
    
    return drug_names, efficacies, rewards, filepath


# ═══════════════════════════════════════════════════════════════
# DATASET LOADER: USE YOUR OWN CSV
# ═══════════════════════════════════════════════════════════════

def load_dataset_from_csv(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Load a bandit dataset from any CSV file.
    
    Expected CSV format:
        id_column, arm_1_name, arm_2_name, ..., arm_K_name
        0,         reward_1,   reward_2,   ..., reward_K
        1,         reward_1,   reward_2,   ..., reward_K
        ...
    
    First column is an ID (ignored). Remaining columns are arm names.
    Each row contains the reward that WOULD have been received if 
    that arm was selected at that time step.
    
    Returns:
        arm_names: list of arm names (from CSV header)
        rewards: numpy array of shape (n_steps, n_arms)
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        arm_names = header[1:]  # Skip ID column
        
        data = []
        for row in reader:
            data.append([float(x) for x in row[1:]])
    
    rewards = np.array(data)
    print(f"\n  📂 Loaded dataset: {filepath}")
    print(f"     {rewards.shape[0]} time steps × {rewards.shape[1]} arms")
    print(f"     Arms: {arm_names}")
    print(f"     Mean rewards per arm: {np.mean(rewards, axis=0).round(4)}")
    
    return arm_names, rewards


# ═══════════════════════════════════════════════════════════════
# RUN UCB1 ON A REWARD MATRIX (offline evaluation)
# ═══════════════════════════════════════════════════════════════

def run_on_dataset(arm_names, true_means, rewards, title="Dataset"):
    """
    Run UCB1 and ε-Greedy on a pre-generated reward matrix.
    
    This is "offline evaluation" — the rewards are pre-determined.
    At time t, the agent picks arm i, and receives rewards[t, i].
    
    This lets us fairly compare algorithms on the SAME data.
    """
    n_steps, n_arms = rewards.shape
    best_arm = np.argmax(true_means)
    best_mean = true_means[best_arm]
    
    print(f"\n  🎯 Best arm: {arm_names[best_arm]} (mean = {best_mean:.4f})")
    print(f"  ⏱  Running {n_steps} steps...\n")
    
    # Run UCB1
    ucb = UCB1Agent(n_arms)
    ucb_regret = []
    
    for t in range(n_steps):
        arm = ucb.select_arm()
        reward = rewards[t, arm]
        ucb.update(arm, reward)
        ucb_regret.append(best_mean - true_means[arm])
    
    ucb_cum_regret = np.cumsum(ucb_regret)
    
    # Run ε-Greedy
    eg = EpsilonGreedyAgent(n_arms, epsilon=0.1)
    eg_regret = []
    
    for t in range(n_steps):
        arm = eg.select_arm()
        reward = rewards[t, arm]
        eg.update(arm, reward)
        eg_regret.append(best_mean - true_means[arm])
    
    eg_cum_regret = np.cumsum(eg_regret)
    
    # Print results
    print(f"  {'─'*50}")
    print(f"  {'Algorithm':<20} {'Total Reward':>12} {'Cum. Regret':>12}")
    print(f"  {'─'*50}")
    print(f"  {'UCB1':<20} {sum(ucb.history_rewards):>12.1f} {ucb_cum_regret[-1]:>12.1f}")
    print(f"  {'ε-Greedy (ε=0.1)':<20} {sum(eg.history_rewards):>12.1f} {eg_cum_regret[-1]:>12.1f}")
    print(f"  {'─'*50}")
    
    # Show arm selection breakdown
    print(f"\n  UCB1 arm pulls:")
    for i in range(n_arms):
        bar = "█" * int(ucb.counts[i] / n_steps * 50)
        pct = ucb.counts[i] / n_steps * 100
        marker = " ⭐" if i == best_arm else ""
        print(f"    {arm_names[i]:<25} {ucb.counts[i]:>5} pulls ({pct:5.1f}%) {bar}{marker}")
    
    print(f"\n  UCB1 learned means vs true means:")
    for i in range(n_arms):
        learned = ucb.get_sample_means()[i]
        true = true_means[i]
        error = abs(learned - true)
        print(f"    {arm_names[i]:<25} Learned: {learned:.4f}  True: {true:.4f}  Error: {error:.4f}")
    
    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cumulative regret
    ax = axes[0]
    steps = np.arange(1, n_steps + 1)
    ax.plot(steps, ucb_cum_regret, color='#2ecc71', linewidth=2, label='UCB1')
    ax.plot(steps, eg_cum_regret, color='#e74c3c', linewidth=2, label='ε-Greedy')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title(f'{title}\nCumulative Regret', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Arm selection distribution
    ax = axes[1]
    x = np.arange(n_arms)
    width = 0.35
    ax.bar(x - width/2, ucb.counts, width, label='UCB1', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, eg.counts, width, label='ε-Greedy', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Number of Pulls')
    ax.set_title('Arm Selection Count', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('_')[0][:8] for n in arm_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Learned vs true means
    ax = axes[2]
    ucb_means = ucb.get_sample_means()
    ax.bar(x - width/2, ucb_means, width, label='UCB1 Estimate', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, true_means, width, label='True Mean', color='#e67e22', alpha=0.8)
    ax.set_xlabel('Arm')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Learned vs True Means', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split('_')[0][:8] for n in arm_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'🎰 UCB1 on {title}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    safe_title = title.lower().replace(' ', '_').replace(':', '')
    save_path = os.path.join(os.path.dirname(__file__), f"result_{safe_title}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved plot: {save_path}")
    plt.close()
    
    return ucb, eg


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   🎰  UCB1 on Real-World Datasets                     🎰  ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║   Dataset 1: Online Ad CTR Optimization (10 ads)           ║
    ║   Dataset 2: Movie Recommendation (8 genres)               ║
    ║   Dataset 3: Clinical Trial Drug Selection (5 drugs)       ║
    ║                                                            ║
    ║   Each generates a CSV you can inspect and reuse!          ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # ── Dataset 1: Online Ads ──
    print("\n" + "=" * 60)
    print("  📊 DATASET 1: Online Ad Click-Through Rate Optimization")
    print("=" * 60)
    ad_names, ctrs, ad_rewards, _ = generate_ad_dataset(n_samples=10000)
    run_on_dataset(ad_names, ctrs, ad_rewards, "Ad CTR Optimization")
    
    # ── Dataset 2: Movies ──
    print("\n" + "=" * 60)
    print("  🎬 DATASET 2: Movie Genre Recommendation")
    print("=" * 60)
    genre_names, genre_means, movie_rewards, _ = generate_movie_dataset(n_users=5000)
    run_on_dataset(genre_names, genre_means, movie_rewards, "Movie Recommendation")
    
    # ── Dataset 3: Clinical Trial ──
    print("\n" + "=" * 60)
    print("  💊 DATASET 3: Clinical Trial Drug Selection")
    print("=" * 60)
    drug_names, efficacies, drug_rewards, _ = generate_clinical_dataset(n_patients=2000)
    run_on_dataset(drug_names, efficacies, drug_rewards, "Clinical Trial")
    
    # ── Show how to load from CSV ──
    print("\n" + "=" * 60)
    print("  📂 BONUS: Loading from CSV (replay the ad dataset)")
    print("=" * 60)
    csv_path = os.path.join(os.path.dirname(__file__), "dataset_ads.csv")
    loaded_names, loaded_rewards = load_dataset_from_csv(csv_path)
    loaded_means = list(np.mean(loaded_rewards, axis=0))
    run_on_dataset(loaded_names, loaded_means, loaded_rewards, "CSV Replay")
    
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║                    ALL DATASETS COMPLETE!                   ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║   Generated CSV files you can inspect:                     ║
    ║     • dataset_ads.csv       (10 ads × 10000 steps)         ║
    ║     • dataset_movies.csv    (8 genres × 5000 users)        ║
    ║     • dataset_clinical.csv  (5 drugs × 2000 patients)      ║
    ║                                                            ║
    ║   Generated result plots:                                  ║
    ║     • result_ad_ctr_optimization.png                       ║
    ║     • result_movie_recommendation.png                      ║
    ║     • result_clinical_trial.png                            ║
    ║     • result_csv_replay.png                                ║
    ║                                                            ║
    ║   💡 You can also load YOUR OWN CSV:                       ║
    ║      arm_names, rewards = load_dataset_from_csv("my.csv")  ║
    ║      means = list(np.mean(rewards, axis=0))                ║
    ║      run_on_dataset(arm_names, means, rewards, "My Data")  ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
