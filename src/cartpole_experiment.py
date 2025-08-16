#!/usr/bin/env python3
"""
cartpole_experiment.py

Comprehensive CartPole RL analysis using bettermdptools' discretization infrastructure.
Compares Value Iteration, Policy Iteration, SARSA, and Q-Learning with different discretization strategies.
"""
import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.utils.test_env import TestEnv


def plot_discretization_analysis(results, outdir):
    """Plot analysis of different discretization granularities."""
    df = pd.DataFrame(results)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Group by discretization (bins) and algorithm
    algorithms = df['algorithm'].unique()
    bins_values = sorted(df['n_bins'].unique())

    colors = {'VI': 'green', 'PI': 'orange', 'SARSA': 'blue', 'Q-Learning': 'red'}

    # 1. Average reward vs discretization
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        alg_grouped = alg_data.groupby('n_bins').agg({
            'avg_reward': ['mean', 'std'],
            'avg_length': ['mean', 'std'],
            'success_rate': ['mean', 'std']
        }).reset_index()

        # Flatten column names
        alg_grouped.columns = ['n_bins', 'reward_mean', 'reward_std', 'length_mean', 'length_std', 'success_mean',
                               'success_std']

        ax1.errorbar(alg_grouped['n_bins'], alg_grouped['reward_mean'],
                     yerr=alg_grouped['reward_std'], label=alg, color=colors[alg],
                     marker='o', capsize=5, linewidth=2)

        ax2.errorbar(alg_grouped['n_bins'], alg_grouped['length_mean'],
                     yerr=alg_grouped['length_std'], label=alg, color=colors[alg],
                     marker='o', capsize=5, linewidth=2)

        ax3.errorbar(alg_grouped['n_bins'], alg_grouped['success_mean'],
                     yerr=alg_grouped['success_std'], label=alg, color=colors[alg],
                     marker='o', capsize=5, linewidth=2)

    ax1.set_xlabel('Discretization Bins (per dimension)')
    ax1.set_ylabel('Average Episode Reward')
    ax1.set_title('Performance vs Discretization Granularity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    ax2.set_xlabel('Discretization Bins (per dimension)')
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Episode Length vs Discretization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    ax3.set_xlabel('Discretization Bins (per dimension)')
    ax3.set_ylabel('Success Rate (Length ≥ 195)')
    ax3.set_title('Success Rate vs Discretization')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)

    # 4. State space size vs performance
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        ax4.scatter(alg_data['state_space_size'], alg_data['avg_reward'],
                    label=alg, color=colors[alg], alpha=0.7, s=60)

    ax4.set_xlabel('State Space Size')
    ax4.set_ylabel('Average Episode Reward')
    ax4.set_title('Performance vs State Space Complexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/cartpole_discretization_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curves(all_learning_data, outdir):
    """Plot learning curves for different discretizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    discretizations = sorted(all_learning_data.keys())
    colors = {'SARSA': 'blue', 'Q-Learning': 'red'}

    for i, n_bins in enumerate(discretizations):
        if i >= 6:  # Limit to 6 subplots
            break

        ax = axes[i]

        for alg in ['SARSA', 'Q-Learning']:
            if alg in all_learning_data[n_bins]:
                rewards = all_learning_data[n_bins][alg]['episode_rewards']
                episodes = np.arange(len(rewards))

                # Plot raw rewards with transparency
                ax.plot(episodes, rewards, alpha=0.3, color=colors[alg])

                # Smooth with rolling average
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else len(rewards)
                if len(rewards) >= window and window > 1:
                    rewards_smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()
                    ax.plot(episodes, rewards_smooth, color=colors[alg], linewidth=2, label=f'{alg} (smooth)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title(f'Learning Curves: {n_bins} bins/dim\n({n_bins ** 4:,} states)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add horizontal line at success threshold
        ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, label='Success threshold')

    # Hide unused subplots
    for i in range(len(discretizations), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/cartpole_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison(V_track_vi, V_track_pi, outdir):
    """Plot VI vs PI convergence comparison."""
    vi_deltas = np.max(np.abs(np.diff(V_track_vi, axis=0)), axis=1)
    pi_deltas = np.max(np.abs(np.diff(V_track_pi, axis=0)), axis=1)
    x_vi = np.arange(1, len(vi_deltas) + 1)
    x_pi = np.arange(1, len(pi_deltas) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Combined, but only up to PI's last iteration
    min_len = min(len(vi_deltas), len(pi_deltas))
    ax1.semilogy(x_vi[:min_len], vi_deltas[:min_len],
                 'b-', label='Value Iteration', alpha=0.8)
    ax1.semilogy(x_pi[:min_len], pi_deltas[:min_len], 'r-', label='Policy Iteration', alpha=0.8)
    ax1.set_title('Convergence Comparison')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Value Change (log scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Set proper x-axis ticks and limits
    if min_len > 1:
        ticks = np.linspace(1, min_len, min(6, min_len), dtype=int)
        ax1.set_xticks(ticks)
        ax1.set_xlim(1, min_len)

    # 2) VI detail (full range)
    ax2.semilogy(x_vi, vi_deltas, 'b-', label='Value Iteration')
    ax2.set_title(f'Value Iteration Detail\n({len(vi_deltas)} iters)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max Value Change (log scale)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if len(vi_deltas) > 1:
        ticks_vi = np.linspace(1, len(vi_deltas), min(6, len(vi_deltas)), dtype=int)
        ax2.set_xticks(ticks_vi)
        ax2.set_xlim(1, len(vi_deltas))

    # 3) PI detail (full range)
    ax3.semilogy(x_pi, pi_deltas, 'r-', label='Policy Iteration')
    ax3.set_title(f'Policy Iteration Detail\n({len(pi_deltas)} iters)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Max Value Change (log scale)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    if len(pi_deltas) > 1:
        ticks_pi = np.linspace(1, len(pi_deltas), min(6, len(pi_deltas)), dtype=int)
        ax3.set_xticks(ticks_pi)
        ax3.set_xlim(1, len(pi_deltas))

    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/cartpole_convergence_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_heatmaps(best_policies, discretizers, outdir):
    """Plot policy heatmaps for different discretizations and algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    discretizations = sorted(best_policies.keys())
    colors = {'VI': 'green', 'PI': 'orange', 'SARSA': 'blue', 'Q-Learning': 'red'}

    for i, n_bins in enumerate(discretizations[:4]):  # Show up to 4 discretizations
        ax = axes[i]

        # Get the best policy for this discretization
        best_alg = best_policies[n_bins]['algorithm']
        policy = best_policies[n_bins]['policy']
        discretizer_info = discretizers[n_bins]

        # Create policy grid for position vs angle (most important dimensions)
        pos_bins = discretizer_info['position_bins']
        angle_bins = discretizer_info['angle_bins']

        # Sample states and extract policy actions
        policy_grid = np.full((angle_bins, pos_bins), -1)  # -1 indicates no data

        # Sample a subset of states for visualization (to avoid memory issues)
        n_samples = min(1000, discretizer_info['n_states'])
        state_indices = np.random.choice(discretizer_info['n_states'], n_samples, replace=False)

        for state_idx in state_indices:
            # Convert state index to component indices using the same logic as DiscretizedCartPole
            # This is a simplified version that assumes uniform binning
            pos_idx = (state_idx // (discretizer_info['velocity_bins'] * angle_bins * discretizer_info[
                'angular_velocity_bins'])) % pos_bins
            angle_idx = (state_idx // (discretizer_info['angular_velocity_bins'])) % angle_bins

            # Get policy action
            if callable(policy):
                action = policy(state_idx)
            else:
                action = policy[state_idx]

            # Store in grid (position vs angle)
            policy_grid[angle_idx, pos_idx] = action

        # Create masked array for visualization
        masked_grid = np.ma.masked_where(policy_grid == -1, policy_grid)

        # Plot heatmap
        im = ax.imshow(masked_grid, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1, origin='lower')

        # Set labels and title
        ax.set_xlabel('Cart Position Bin')
        ax.set_ylabel('Pole Angle Bin')
        ax.set_title(f'{best_alg} Policy ({n_bins} bins/dim)\n0=Left, 1=Right')

        # Set tick labels
        ax.set_xticks(range(0, pos_bins, max(1, pos_bins // 5)))
        ax.set_xticklabels([f'{i}' for i in range(0, pos_bins, max(1, pos_bins // 5))])
        ax.set_yticks(range(0, angle_bins, max(1, angle_bins // 5)))
        ax.set_yticklabels([f'{i}' for i in range(0, angle_bins, max(1, angle_bins // 5))])

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Action (0=Left, 1=Right)')

        # Add performance info
        performance = best_policies[n_bins]['performance']
        ax.text(0.02, 0.98, f'Reward: {performance:.1f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for i in range(len(discretizations), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/cartpole_policy_heatmaps.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(final_results, outdir):
    """Plot comprehensive algorithm comparison."""
    df = pd.DataFrame(final_results)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Performance comparison across discretizations
    algorithms = df['algorithm'].unique()
    discretizations = sorted(df['n_bins'].unique())

    width = 0.2
    x = np.arange(len(discretizations))

    for i, alg in enumerate(algorithms):
        alg_data = df[df['algorithm'] == alg]
        grouped = alg_data.groupby('n_bins')['avg_reward'].agg(['mean', 'std']).reset_index()

        ax1.bar(x + i * width, grouped['mean'], width, yerr=grouped['std'],
                label=alg, alpha=0.7, capsize=5)

    ax1.set_xlabel('Discretization Bins (per dimension)')
    ax1.set_ylabel('Average Episode Reward')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(discretizations)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Success rate comparison
    for i, alg in enumerate(algorithms):
        alg_data = df[df['algorithm'] == alg]
        grouped = alg_data.groupby('n_bins')['success_rate'].agg(['mean', 'std']).reset_index()

        ax2.bar(x + i * width, grouped['mean'], width, yerr=grouped['std'],
                label=alg, alpha=0.7, capsize=5)

    ax2.set_xlabel('Discretization Bins (per dimension)')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate Comparison')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(discretizations)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Convergence time (only for VI/PI)
    converged = df[df['training_time'].notna()]
    if not converged.empty:
        conv_algorithms = converged['algorithm'].unique()
        conv_x_pos = np.arange(len(conv_algorithms))
        conv_times = [converged[converged['algorithm'] == alg]['training_time'].mean() for alg in conv_algorithms]

        ax3.bar(conv_x_pos, conv_times, alpha=0.7, color='lightgreen')
        ax3.set_ylabel('Wall Time (seconds)')
        ax3.set_title('Convergence Time Comparison')
        ax3.set_xticks(conv_x_pos)
        ax3.set_xticklabels(conv_algorithms)
        ax3.grid(True, alpha=0.3)

    # 4. Best performance summary
    best_results = df.loc[df.groupby('algorithm')['avg_reward'].idxmax()]

    colors = {'VI': 'green', 'PI': 'orange', 'SARSA': 'blue', 'Q-Learning': 'red'}
    bar_colors = [colors[alg] for alg in best_results['algorithm']]

    ax4.bar(best_results['algorithm'], best_results['avg_reward'],
            alpha=0.7, color=bar_colors)

    for i, row in best_results.iterrows():
        ax4.annotate(f'{row["n_bins"]} bins\n{row["avg_reward"]:.1f} reward',
                     (row['algorithm'], row['avg_reward']),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=10)

    ax4.set_ylabel('Best Average Episode Reward')
    ax4.set_title('Best Performance by Algorithm')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/cartpole_algorithm_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_model_based_experiment(env, n_bins, seed, gamma=0.99, theta=1e-6):
    """Run Value Iteration and Policy Iteration on discretized CartPole."""
    planner = Planner(env.P)

    # Value Iteration
    t0 = time.time()
    V_vi, V_track_vi, pi_vi = planner.value_iteration(gamma=gamma, theta=theta)
    vi_time = time.time() - t0
    vi_iters = len(V_track_vi)

    # Policy Iteration
    t0 = time.time()
    V_pi, V_track_pi, pi_pi = planner.policy_iteration(gamma=gamma, theta=theta)
    pi_time = time.time() - t0
    pi_iters = len(V_track_pi)

    # Test policies using TestEnv
    vi_scores = TestEnv.test_env(env=env, n_iters=200, pi=pi_vi)
    pi_scores = TestEnv.test_env(env=env, n_iters=200, pi=pi_pi)

    vi_reward = np.mean(vi_scores)
    pi_reward = np.mean(pi_scores)

    # For CartPole, success is typically defined as episode length >= 195
    vi_success = np.mean(vi_scores >= 195)
    pi_success = np.mean(pi_scores >= 195)

    # Average length is the same as average reward for CartPole (1 reward per step)
    vi_length = vi_reward
    pi_length = pi_reward


    return {
        'VI': {
            'V': V_vi, 'V_track': V_track_vi, 'policy': pi_vi, 'iters': vi_iters,
            'training_time': vi_time, 'avg_reward': vi_reward, 'avg_length': vi_length,
            'success_rate': vi_success
        },
        'PI': {
            'V': V_pi, 'V_track': V_track_pi, 'policy': pi_pi, 'iters': pi_iters,
            'training_time': pi_time, 'avg_reward': pi_reward, 'avg_length': pi_length,
            'success_rate': pi_success
        }
    }


def run_model_free_experiment(env, algorithm, n_episodes=1000, seed=None):
    """Run SARSA or Q-Learning on discretized CartPole."""
    agent = RL(env)

    if seed is not None:
        np.random.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)

    start_time = time.time()

    if algorithm == 'SARSA':
        Q, V, policy, Q_track, pi_track, episode_rewards = agent.sarsa(
            gamma=0.99,
            n_episodes=n_episodes,
            init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
            init_epsilon=1.0, min_epsilon=0.05, epsilon_decay_ratio=0.9,
        )
    elif algorithm == 'Q-Learning':
        Q, V, policy, Q_track, pi_track, episode_rewards = agent.q_learning(
            gamma=0.99,
            n_episodes=n_episodes,
            init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
            init_epsilon=1.0, min_epsilon=0.05, epsilon_decay_ratio=0.9,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    training_time = time.time() - start_time

    # Test final policy using TestEnv
    test_scores = TestEnv.test_env(env=env, n_iters=100, pi=policy)
    avg_reward = np.mean(test_scores)
    success_rate = np.mean(test_scores >= 195)
    avg_length = avg_reward  # For CartPole, length = reward (1 reward per step)

    # Find convergence episode (when 50-episode rolling average reaches 195)
    convergence_episode = None
    if len(episode_rewards) >= 50:
        rolling_avg = pd.Series(episode_rewards).rolling(50).mean()
        convergence_idx = np.where(rolling_avg >= 195)[0]
        if len(convergence_idx) > 0:
            convergence_episode = convergence_idx[0] + 50

    return {
        'Q': Q, 'V': V, 'policy': policy, 'episode_rewards': episode_rewards,
        'training_time': training_time, 'avg_reward': avg_reward, 'avg_length': avg_length,
        'success_rate': success_rate, 'convergence_episode': convergence_episode
    }


def main(args):
    outdir = os.path.abspath(args.outdir)
    os.makedirs(f"{outdir}/results", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)

    print("=== CartPole RL Comprehensive Analysis ===")
    print("Using bettermdptools CartpoleWrapper for discretization\n")

    # Discretization configurations to test
    discretizations = [3, 6]
    algorithms = ['VI', 'PI', 'SARSA', 'Q-Learning']
    n_seeds = args.n_seeds

    print(f"Testing {len(discretizations)} discretizations × {len(algorithms)} algorithms × {n_seeds} seeds")
    print(f"Discretizations: {discretizations} bins per dimension")
    print(f"Episodes per run: {args.episodes}\n")

    # Store all results
    all_results = []
    all_learning_data = {}
    convergence_data = {}
    best_policies = {}
    discretizers = {}

    # Run experiments
    for n_bins in discretizations:
        print(f"\n=== Discretization: {n_bins} bins per dimension ({n_bins ** 4:,} total states) ===")

        # Create discretized environment
        raw_env = gym.make("CartPole-v1")
        env = CartpoleWrapper(
            raw_env,
            position_bins=n_bins,
            velocity_bins=n_bins,
            angular_velocity_bins=n_bins,
            threshold_bins=0.5,
            angular_center_resolution=0.1,
            angular_outer_resolution=0.5
        )

        # Store discretizer info for visualization
        discretizers[n_bins] = {
            'position_bins': n_bins,
            'velocity_bins': n_bins,
            'angular_velocity_bins': n_bins,
            'angle_bins': env.observation_space.n // (n_bins ** 3),  # Approximate
            'n_states': env.observation_space.n
        }

        print(f"Created discrete environment with {env.observation_space.n:,} states")

        # Model-based methods (VI and PI)
        if 'VI' in algorithms or 'PI' in algorithms:
            print("\nRunning model-based methods...")

            for seed in range(n_seeds):
                print(f"  Seed {seed + 1}/{n_seeds}...", end=' ')

                mb_results = run_model_based_experiment(env, n_bins, seed)

                # Store VI results
                if 'VI' in algorithms:
                    vi_result = {
                        'algorithm': 'VI',
                        'n_bins': n_bins,
                        'state_space_size': env.observation_space.n,
                        'seed': seed,
                        'iters': mb_results['VI']['iters'],
                        'training_time': mb_results['VI']['training_time'],
                        'avg_reward': mb_results['VI']['avg_reward'],
                        'avg_length': mb_results['VI']['avg_length'],
                        'success_rate': mb_results['VI']['success_rate']
                    }
                    all_results.append(vi_result)

                    # Track best policy for this discretization
                    if n_bins not in best_policies or mb_results['VI']['avg_reward'] > best_policies[n_bins][
                        'performance']:
                        best_policies[n_bins] = {
                            'policy': mb_results['VI']['policy'],
                            'performance': mb_results['VI']['avg_reward'],
                            'algorithm': 'VI'
                        }

                    # Store convergence data for first seed
                    if seed == 0:
                        convergence_data[f'VI_{n_bins}'] = mb_results['VI']['V_track']

                # Store PI results
                if 'PI' in algorithms:
                    pi_result = {
                        'algorithm': 'PI',
                        'n_bins': n_bins,
                        'state_space_size': env.observation_space.n,
                        'seed': seed,
                        'iters': mb_results['PI']['iters'],
                        'training_time': mb_results['PI']['training_time'],
                        'avg_reward': mb_results['PI']['avg_reward'],
                        'avg_length': mb_results['PI']['avg_length'],
                        'success_rate': mb_results['PI']['success_rate']
                    }
                    all_results.append(pi_result)

                    # Track best policy for this discretization
                    if n_bins not in best_policies or mb_results['PI']['avg_reward'] > best_policies[n_bins][
                        'performance']:
                        best_policies[n_bins] = {
                            'policy': mb_results['PI']['policy'],
                            'performance': mb_results['PI']['avg_reward'],
                            'algorithm': 'PI'
                        }

                    # Store convergence data for first seed
                    if seed == 0:
                        convergence_data[f'PI_{n_bins}'] = mb_results['PI']['V_track']

                print(f"VI: {mb_results['VI']['avg_reward']:.1f}, PI: {mb_results['PI']['avg_reward']:.1f}")

        # Model-free methods (SARSA and Q-Learning)
        if 'SARSA' in algorithms or 'Q-Learning' in algorithms:
            print("\nRunning model-free methods...")

            all_learning_data[n_bins] = {}

            for alg in ['SARSA', 'Q-Learning']:
                if alg in algorithms:
                    print(f"\nRunning {alg}...")

                    for seed in range(n_seeds):
                        print(f"  Seed {seed + 1}/{n_seeds}...", end=' ')

                        result = run_model_free_experiment(env, alg, args.episodes, seed)

                        # Store individual result
                        individual_result = {
                            'algorithm': alg,
                            'n_bins': n_bins,
                            'state_space_size': env.observation_space.n,
                            'seed': seed,
                            'training_time': result['training_time'],
                            'avg_reward': result['avg_reward'],
                            'avg_length': result['avg_length'],
                            'success_rate': result['success_rate'],
                            'convergence_episode': result['convergence_episode']
                        }

                        all_results.append(individual_result)

                        # Track best policy for this discretization
                        if n_bins not in best_policies or result['avg_reward'] > best_policies[n_bins]['performance']:
                            best_policies[n_bins] = {
                                'policy': result['policy'],
                                'performance': result['avg_reward'],
                                'algorithm': alg
                            }

                        # Store learning data for first seed
                        if seed == 0:
                            all_learning_data[n_bins][alg] = {
                                'episode_rewards': result['episode_rewards']
                            }

                        print(f"Reward: {result['avg_reward']:.1f}, Success: {result['success_rate']:.2f}")

        # Close environment
        raw_env.close()

    # Save results with rounding
    results_df = pd.DataFrame(all_results)
    # Round all numeric columns to 4 decimal places
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    results_df.to_csv(f"{outdir}/results/cartpole_comprehensive_results.csv", index=False)
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")

    plot_discretization_analysis(all_results, outdir)
    print(f"✓ Discretization analysis saved to: {outdir}/figures/cartpole_discretization_analysis.png")

    if all_learning_data:
        plot_learning_curves(all_learning_data, outdir)
        print(f"✓ Learning curves saved to: {outdir}/figures/cartpole_learning_curves.png")

    if convergence_data:
        # Plot convergence for the finest discretization
        finest_bins = max(discretizations)
        if f'VI_{finest_bins}' in convergence_data and f'PI_{finest_bins}' in convergence_data:
            plot_convergence_comparison(convergence_data[f'VI_{finest_bins}'],
                                        convergence_data[f'PI_{finest_bins}'], outdir)
            print(f"✓ Convergence comparison saved to: {outdir}/figures/cartpole_convergence_comparison.png")

    plot_algorithm_comparison(all_results, outdir)
    print(f"✓ Algorithm comparison saved to: {outdir}/figures/cartpole_algorithm_comparison.png")

    # Generate policy heatmaps if we have best policies
    if best_policies:
        plot_policy_heatmaps(best_policies, discretizers, outdir)
        print(f"✓ Policy heatmaps saved to: {outdir}/figures/cartpole_policy_heatmaps.png")

    # Print final analysis
    print("\n=== Final Analysis ===")

    # Best overall performance
    best_overall = results_df.loc[results_df['avg_reward'].idxmax()]
    print(f"Best Performance: {best_overall['algorithm']} with {best_overall['n_bins']} bins")
    print(f"  Reward: {best_overall['avg_reward']:.1f}")
    print(f"  Success Rate: {best_overall['success_rate']:.2f}")
    print(f"  State Space: {best_overall['state_space_size']:,} states")

    # Algorithm comparison
    alg_summary = results_df.groupby('algorithm').agg({
        'avg_reward': ['mean', 'std'],
        'success_rate': ['mean', 'std'],
        'training_time': ['mean', 'std']
    }).round(3)

    print(f"\nAlgorithm Summary:")
    print(alg_summary)

    # Discretization impact
    disc_summary = results_df.groupby('n_bins').agg({
        'avg_reward': ['mean', 'std'],
        'success_rate': ['mean', 'std'],
        'state_space_size': 'first'
    }).round(3)

    print(f"\nDiscretization Impact:")
    print(disc_summary)

    print(f"\n✓ Results saved to: {outdir}/results/cartpole_comprehensive_results.csv")
    print("✓ CartPole comprehensive analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole RL Comprehensive Analysis")
    parser.add_argument("--episodes", type=int, default=20000, help="Episodes per experiment")
    parser.add_argument("--n_seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--outdir", type=str, default="..", help="Output directory")

    args = parser.parse_args()

    # Set base seed for reproducibility
    np.random.seed(args.seed)

    main(args)
