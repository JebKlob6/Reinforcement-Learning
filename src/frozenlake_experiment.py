# frozenlake_minimal_example.py
"""Minimal, step-by-step demo of using bettermdptools on FrozenLake-v1
---------------------------------------------------------------------------
✓  Builds the Gym env (deterministic 4×4 map).
✓  Wraps it with `Planner` to use transition P tables.
✓  Runs Value Iteration (VI), Policy Iteration (PI), and SARSA from bettermdptools.
✓  Prints convergence metrics so you can answer "compare convergence rates?"
✓  Generates comprehensive plots for analysis
"""

import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Use bettermdptools implementations
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL

# Create results and figures directories
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def plot_convergence_comparison(V_track_vi, V_track_pi, filename=None):
    """Plot VI vs PI convergence comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    deltas_vi = [np.max(np.abs(V_track_vi[i] - V_track_vi[i - 1])) for i in range(1, len(V_track_vi))]
    deltas_pi = [np.max(np.abs(V_track_pi[i] - V_track_pi[i - 1])) for i in range(1, len(V_track_pi))]

    ax1.semilogy(deltas_vi, 'b-', label='Value Iteration', linewidth=2)
    ax1.semilogy(deltas_pi, 'r-', label='Policy Iteration', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Value Change (log scale)')
    ax1.set_title('Convergence Rate Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(np.arange(len(V_track_vi[-1])) - 0.2, V_track_vi[-1], 0.4, label='Value Iteration', alpha=0.7)
    ax2.bar(np.arange(len(V_track_pi[-1])) + 0.2, V_track_pi[-1], 0.4, label='Policy Iteration', alpha=0.7)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Value')
    ax2.set_title('Final Value Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def plot_policy_visualization(policy, env_size=4, title="Policy", filename=None):
    """Visualize a discrete policy for FrozenLake."""
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    if callable(policy):
        policy_grid = np.array([action_symbols[policy(s)] for s in range(env_size * env_size)]).reshape(env_size,
                                                                                                        env_size)
    else:
        policy_grid = np.array([action_symbols[a] for a in policy]).reshape(env_size, env_size)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(env_size + 1):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)

    for i in range(env_size):
        for j in range(env_size):
            ax.text(j + 0.5, env_size - i - 0.5, policy_grid[i, j], ha='center', va='center', fontsize=24,
                    fontweight='bold')
    ax.text(0.5, env_size - 0.5, 'S', ha='center', va='center', fontsize=16, color='green', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax.text(env_size - 0.5, 0.5, 'G', ha='center', va='center', fontsize=16, color='red', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    for hole in [5, 7, 11, 12]:
        r, c = divmod(hole, env_size)
        ax.add_patch(plt.Rectangle((c, env_size - r - 1), 1, 1, facecolor='lightblue', alpha=0.5))
        ax.text(c + 0.5, env_size - r - 0.5, 'H', ha='center', va='center', fontsize=16, color='blue', weight='bold')

    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sarsa_exploration_analysis(sarsa_results, filename=None):
    """Plot SARSA performance for different epsilon strategies."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['blue', 'orange', 'green', 'red']

    for i, (eps, data) in enumerate(sarsa_results.items()):
        rewards = data['episode_rewards']
        smooth = pd.Series(rewards).rolling(50).mean()
        ax1.plot(smooth, label=f'ε = {eps}', color=colors[i], linewidth=2)
        ax2.plot(rewards, alpha=0.3, color=colors[i])
        ax2.plot(smooth, label=f'ε = {eps}', color=colors[i], linewidth=2)
    ax1.set(title='SARSA Learning Curves (50-ep smooth)', xlabel='Episode', ylabel='Avg Reward')
    ax2.set(title='SARSA Raw vs Smoothed Rewards', xlabel='Episode', ylabel='Reward')
    ax1.legend();
    ax2.legend()
    ax1.grid(alpha=0.3);
    ax2.grid(alpha=0.3)

    final_means = [np.mean(d['test_rewards']) for d in sarsa_results.values()]
    final_stds = [np.std(d['test_rewards']) for d in sarsa_results.values()]
    eps_list = list(sarsa_results.keys())
    ax3.bar(range(len(eps_list)), final_means, yerr=final_stds, capsize=5, alpha=0.7, color=colors[:len(eps_list)])
    ax3.set(title='Final Test Performance', xlabel='ε value', ylabel='Mean Reward')
    ax3.set_xticks(range(len(eps_list)));
    ax3.set_xticklabels([f'{e}' for e in eps_list])
    ax3.grid(alpha=0.3)

    for i, (eps, data) in enumerate(sarsa_results.items()):
        decay = data.get('decay_schedule', [eps * (0.9 ** (ep // 100)) for ep in range(len(data['episode_rewards']))])
        ax4.plot(decay, label=f'ε = {eps}', color=colors[i], linewidth=2)
    ax4.set(title='Exploration Decay Over Time', xlabel='Episode', ylabel='ε value')
    ax4.legend();
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def run_sarsa_experiments(env, epsilon_values, n_episodes=1000):
    """Run SARSA from bettermdptools with different exploration strategies."""
    results = {}
    for eps in epsilon_values:
        print(f"Running SARSA (bettermdptools) with ε={eps}")
        agent = RL(env)
        start = time.time()
        Q, V, pi, Q_track, pi_track, episode_rewards = agent.sarsa(
            gamma=0.99,
            init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
            init_epsilon=eps, min_epsilon=0.01, epsilon_decay_ratio=0.9,
            n_episodes=n_episodes
        )
        elapsed = time.time() - start

        # Evaluate final policy
        test_rewards = []
        for _ in range(100):
            state, _ = env.reset()
            total = 0
            for t in range(100):
                if isinstance(pi, dict):
                    action = pi[state]
                else:
                    action = pi(state)
                s, r, done, truncated, _ = env.step(action)
                total += r
                if done or truncated:
                    break
                state = s
            test_rewards.append(total)

        results[eps] = {
            'Q': Q,
            'V': V,
            'pi': pi,
            'episode_rewards': episode_rewards,
            'test_rewards': test_rewards,
            'training_time': elapsed
        }
        print(f"  Time: {elapsed:.2f}s | Test: {np.mean(test_rewards):.3f}±{np.std(test_rewards):.3f}")

    return results


def main():
    print("=== FrozenLake Comprehensive Analysis ===")
    env_det = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    env_stoch = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

    planner = Planner(env_det.unwrapped.P)
    V_vi, V_track_vi, pi_vi = planner.value_iteration(gamma=0.99, theta=1e-6)
    V_pi, V_track_pi, pi_pi = planner.policy_iteration(gamma=0.99, theta=1e-6)

    eps_vals = [0.1, 0.3, 0.5, 0.7]
    sarsa_results = run_sarsa_experiments(env_stoch, eps_vals, n_episodes=1000)

    print(f"VI iters: {len(V_track_vi)} | PI iters: {len(V_track_pi)}")
    for e, d in sarsa_results.items():
        print(f"ε={e} -> {np.mean(d['test_rewards']):.3f}±{np.std(d['test_rewards']):.3f}")

    plot_convergence_comparison(V_track_vi, V_track_pi, filename="figures/frozenlake_convergence.png")
    plot_policy_visualization(pi_vi, title="VI Policy", filename="figures/vi_policy.png")
    plot_policy_visualization(pi_pi, title="PI Policy", filename="figures/pi_policy.png")
    plot_sarsa_exploration_analysis(sarsa_results, filename="figures/sarsa_exploration.png")

    results_df = pd.DataFrame([
        {'alg': 'VI', 'iters': len(V_track_vi), 'time': None},
        {'alg': 'PI', 'iters': len(V_track_pi), 'time': None},
        *[
            {
                'alg': 'SARSA', 'eps': e, 'time': d['training_time'],
                'mean_test': np.mean(d['test_rewards']),
                'std_test': np.std(d['test_rewards'])
            }
            for e, d in sarsa_results.items()
        ]
    ])
    results_df.to_csv('results/frozenlake_results.csv', index=False)

    env_det.close()
    env_stoch.close()
    print("Analysis complete. See results/ and figures/")


if __name__ == '__main__':
    main()
