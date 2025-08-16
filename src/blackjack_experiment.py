import argparse
import os
import time

# local wrapper for Blackjack MDP
import bettermdptools.envs.blackjack_wrapper as BJ
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL
from bettermdptools.utils.test_env import TestEnv


def create_obs_to_state_mapping(raw_env, wrapped_env, num_samples=10000):
    """Create a mapping from raw observations to wrapped states."""
    mapping = {}
    for seed in range(num_samples):
        try:
            raw_obs, _ = raw_env.reset(seed=seed)
            wrapped_state, _ = wrapped_env.reset(seed=seed)
            mapping[raw_obs] = wrapped_state
        except:
            continue
    return mapping


def plot_convergence_comparison(V_track_vi, V_track_pi, outdir):
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
    plt.savefig(f"{outdir}/figures/blackjack_convergence_comparison.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_heatmap(policy, obs_to_state_map, algorithm_name, outdir):
    """Plot policy as a heatmap for any algorithm."""
    # Create reverse mapping from state to observation
    state_to_obs = {v: k for k, v in obs_to_state_map.items()}

    # Create grids for visualization (player sum vs dealer card)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Policy without usable ace
    policy_grid_no_ace = np.full((10, 10), -1)  # -1 indicates no data
    # Policy with usable ace
    policy_grid_ace = np.full((10, 10), -1)

    # Fill grids using actual state mapping
    for state_idx in range(len(policy)):
        if state_idx in state_to_obs:
            obs = state_to_obs[state_idx]
            player_sum, dealer_card, usable_ace = obs

            # Only consider reasonable Blackjack states
            if 12 <= player_sum <= 21 and 1 <= dealer_card <= 10:
                row = player_sum - 12  # 0-9
                col = dealer_card - 1  # 0-9
                action = policy[state_idx]

                if usable_ace:
                    policy_grid_ace[row, col] = action
                else:
                    policy_grid_no_ace[row, col] = action

    # Plot no usable ace
    masked_no_ace = np.ma.masked_where(policy_grid_no_ace == -1, policy_grid_no_ace)
    im1 = ax1.imshow(masked_no_ace, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1, origin='lower')
    ax1.set_title(f'{algorithm_name} Policy (No Usable Ace)\n0=Stick, 1=Hit')
    ax1.set_xlabel('Dealer Card')
    ax1.set_ylabel('Player Sum')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(range(1, 11))
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(range(12, 22))
    plt.colorbar(im1, ax=ax1)

    # Plot with usable ace
    masked_ace = np.ma.masked_where(policy_grid_ace == -1, policy_grid_ace)
    im2 = ax2.imshow(masked_ace, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1, origin='lower')
    ax2.set_title(f'{algorithm_name} Policy (Usable Ace)\n0=Stick, 1=Hit')
    ax2.set_xlabel('Dealer Card')
    ax2.set_ylabel('Player Sum')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels(range(1, 11))
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(range(12, 22))
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/blackjack_{algorithm_name.lower()}_policy.png", dpi=150, bbox_inches='tight')
    plt.close()

    return policy_grid_no_ace, policy_grid_ace


def plot_policy_comparison(pi_vi, pi_pi, pi_sarsa, obs_to_state_map, outdir):
    """Plot comparison of all three policies side by side."""
    # Create reverse mapping from state to observation
    state_to_obs = {v: k for k, v in obs_to_state_map.items()}

    # Initialize grids for all algorithms
    algorithms = ['VI', 'PI', 'SARSA']
    policies = [pi_vi, pi_pi, pi_sarsa]

    # For both ace conditions
    for ace_condition in [False, True]:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ace_label = "Usable Ace" if ace_condition else "No Usable Ace"

        for i, (algorithm, policy) in enumerate(zip(algorithms, policies)):
            policy_grid = np.full((10, 10), -1)  # -1 indicates no data

            # Fill grid using actual state mapping
            for state_idx in range(len(policy)):
                if state_idx in state_to_obs:
                    obs = state_to_obs[state_idx]
                    player_sum, dealer_card, usable_ace = obs

                    # Only consider reasonable Blackjack states and matching ace condition
                    if (12 <= player_sum <= 21 and 1 <= dealer_card <= 10 and
                            usable_ace == ace_condition):
                        row = player_sum - 12  # 0-9
                        col = dealer_card - 1  # 0-9
                        action = policy[state_idx]
                        policy_grid[row, col] = action

            # Plot
            masked_grid = np.ma.masked_where(policy_grid == -1, policy_grid)
            im = axes[i].imshow(masked_grid, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1, origin='lower')
            axes[i].set_title(f'{algorithm} Policy ({ace_label})\n0=Stick, 1=Hit')
            axes[i].set_xlabel('Dealer Card')
            axes[i].set_ylabel('Player Sum')
            axes[i].set_xticks(range(10))
            axes[i].set_xticklabels(range(1, 11))
            axes[i].set_yticks(range(10))
            axes[i].set_yticklabels(range(12, 22))
            plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        ace_suffix = "ace" if ace_condition else "no_ace"
        plt.savefig(f"{outdir}/figures/blackjack_policy_comparison_{ace_suffix}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_sarsa_policy(Q, obs_to_state_map, outdir):
    """Plot SARSA policy as a heatmap."""
    # Extract policy from Q-values
    policy = np.argmax(Q, axis=1)
    plot_policy_heatmap(policy, obs_to_state_map, "SARSA", outdir)


def plot_learning_curves(rewards, outdir=None):
    """Plot learning curves for SARSA."""
    plt.figure(figsize=(12, 6))

    # SARSA learning curve
    plt.subplot(1, 2, 1)
    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Rewards')

    # Smooth with rolling average
    window = min(100, len(rewards) // 10) if len(rewards) > 10 else len(rewards)
    if len(rewards) >= window and window > 1:
        rewards_smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()
        plt.plot(episodes, rewards_smooth, color='blue', linewidth=2, label=f'{window}-Episode Average')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SARSA Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set proper x-axis
    if len(episodes) > 0:
        plt.xlim(0, len(episodes) - 1)
        # Set reasonable number of x-ticks
        max_ticks = min(10, len(episodes))
        if max_ticks > 1:
            tick_positions = np.linspace(0, len(episodes) - 1, max_ticks, dtype=int)
            plt.xticks(tick_positions)

    # SARSA win rate over time
    plt.subplot(1, 2, 2)
    if len(rewards) >= window and window > 1:
        win_rate = pd.Series((np.array(rewards) > 0).astype(float)).rolling(window, min_periods=1).mean()
        plt.plot(episodes, win_rate, color='green', linewidth=2)
        plt.xlim(0, len(episodes) - 1)
        if max_ticks > 1:
            plt.xticks(tick_positions)

    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title(f'SARSA Win Rate ({window}-Episode Window)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if outdir:
        plt.savefig(f"{outdir}/figures/blackjack_learning_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison(results, outdir):
    """Plot comparison of all algorithms."""
    df = pd.DataFrame(results)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Performance comparison (avg reward)
    model_based = df[df['algorithm'].isin(['VI', 'PI'])]
    model_free = df[df['algorithm'].isin(['SARSA'])]

    # Combine for proper x-axis positioning
    algorithms = df['algorithm'].tolist()
    x_pos = np.arange(len(algorithms))
    colors = ['skyblue' if alg in ['VI', 'PI'] else 'lightcoral' for alg in algorithms]

    ax1.bar(x_pos, df['avg_reward'], alpha=0.7, color=colors)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms)
    ax1.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', alpha=0.7, label='Model-Based'),
                       Patch(facecolor='lightcoral', alpha=0.7, label='Model-Free')]
    ax1.legend(handles=legend_elements)

    # Win rate comparison
    ax2.bar(x_pos, df['win_rate'], alpha=0.7, color=colors)
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms)
    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)

    # Convergence time (only for VI/PI)
    converged = df[df['training_time'].notna()]
    if not converged.empty:
        conv_algorithms = converged['algorithm'].tolist()
        conv_x_pos = np.arange(len(conv_algorithms))
        ax3.bar(conv_x_pos, converged['training_time'], alpha=0.7, color='lightgreen')
        ax3.set_ylabel('Wall Time (seconds)')
        ax3.set_title('Convergence Time Comparison')
        ax3.set_xticks(conv_x_pos)
        ax3.set_xticklabels(conv_algorithms)
        ax3.grid(True, alpha=0.3)

    # Iterations to converge (only for VI/PI)
    if not converged.empty:
        ax4.bar(conv_x_pos, converged['iters'], alpha=0.7, color='gold')
        ax4.set_ylabel('Iterations')
        ax4.set_title('Iterations to Convergence')
        ax4.set_xticks(conv_x_pos)
        ax4.set_xticklabels(conv_algorithms)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outdir}/figures/blackjack_algorithm_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main(args):
    outdir = os.path.abspath(args.outdir)
    os.makedirs(f"{outdir}/results", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)

    # Seed control
    np.random.seed(args.seed)
    raw = gym.make("Blackjack-v1")
    raw.reset(seed=args.seed)
    raw.action_space.seed(args.seed)

    # wrap into deterministic MDP
    env = BJ.BlackjackWrapper(raw)
    mdp = env  # now env.P holds transition table

    planner = Planner(mdp.P)
    agent = RL(env)  # Use wrapped env instead of raw

    # Create mapping from raw observations to wrapped states for VI/PI evaluation
    print("Creating observation-to-state mapping...")
    obs_to_state_map = create_obs_to_state_mapping(raw, env, num_samples=10000)
    print(f"Created mapping for {len(obs_to_state_map)} unique observations")

    results = []
    gamma, theta = 1.0, 1e-6

    # Value Iteration
    t0 = time.time()
    V_vi, V_track_vi, pi_vi = planner.value_iteration(gamma=gamma, theta=theta)
    vi_time = time.time() - t0
    vi_iters = len(V_track_vi)

    scores_vi = TestEnv.test_env(
        env=env,
        n_iters=2000,
        pi=pi_vi,
    )

    vi_reward = np.mean(scores_vi)
    vi_win = np.mean(scores_vi > 0)


    results.append(
        {"algorithm": "VI", "iters": vi_iters, "training_time": vi_time, "avg_reward": vi_reward, "win_rate": vi_win})
    pd.DataFrame({"delta": np.max(np.abs(np.diff(V_track_vi, axis=0)), axis=1)}).to_csv(
        f"{outdir}/results/vi_deltas.csv", index=False)

    # Policy Iteration
    t0 = time.time()
    V_pi, V_track_pi, pi_pi = planner.policy_iteration(gamma=gamma, theta=theta)
    pi_time = time.time() - t0
    pi_iters = len(V_track_pi)
    scores_vi = TestEnv.test_env(
        env=env,
        n_iters=2000,
        pi=pi_pi,
    )

    pi_reward = np.mean(scores_vi)
    pi_win = np.mean(scores_vi > 0)

    results.append(
        {"algorithm": "PI", "iters": pi_iters, "training_time": pi_time, "avg_reward": pi_reward, "win_rate": pi_win})

    # SARSA
    t0_sarsa = time.time()
    Q, _, pi_sarsa, _, _, rewards = agent.sarsa(
        gamma=gamma,
        n_episodes=args.episodes,
        init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,
        init_epsilon=1.0, min_epsilon=0.05, epsilon_decay_ratio=0.9,
    )
    sarsa_time = time.time() - t0_sarsa
    sarsa_reward = rewards[-100:].mean()
    sarsa_win = (rewards[-100:] > 0).mean()
    pd.DataFrame({"episode_reward": rewards}).to_csv(f"{outdir}/results/sarsa_rewards.csv", index=False)
    results.append(
        {"algorithm": "SARSA", "iters": len(rewards), "training_time": sarsa_time, "avg_reward": sarsa_reward,
         "win_rate": sarsa_win})

    # Extract SARSA policy from Q-values for comparison
    pi_sarsa = np.argmax(Q, axis=1)

    # Save summary with rounding
    results_df = pd.DataFrame(results)
    # Round all numeric columns to 4 decimal places
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    results_df.to_csv(f"{outdir}/results/blackjack_summary.csv", index=False)

    # Generate comprehensive visualizations
    print("\n=== Generating Visualizations ===")
    plot_convergence_comparison(V_track_vi, V_track_pi, outdir)

    # Individual policy heatmaps for each algorithm
    plot_policy_heatmap(pi_vi, obs_to_state_map, "VI", outdir)
    plot_policy_heatmap(pi_pi, obs_to_state_map, "PI", outdir)
    plot_policy_heatmap(pi_sarsa, obs_to_state_map, "SARSA", outdir)

    # Side-by-side policy comparison
    plot_policy_comparison(pi_vi, pi_pi, pi_sarsa, obs_to_state_map, outdir)
    
    plot_learning_curves(rewards, outdir)
    plot_algorithm_comparison(results, outdir)

    print(f"✓ Convergence comparison saved to: {outdir}/figures/blackjack_convergence_comparison.png")
    print(f"✓ VI policy heatmap saved to: {outdir}/figures/blackjack_vi_policy.png")
    print(f"✓ PI policy heatmap saved to: {outdir}/figures/blackjack_pi_policy.png")
    print(f"✓ SARSA policy heatmap saved to: {outdir}/figures/blackjack_sarsa_policy.png")
    print(f"✓ Policy comparison saved to: {outdir}/figures/blackjack_policy_comparison_ace.png")
    print(f"✓ Policy comparison saved to: {outdir}/figures/blackjack_policy_comparison_no_ace.png")
    print(f"✓ Learning curves saved to: {outdir}/figures/blackjack_learning_curves.png")
    print(f"✓ Algorithm comparison saved to: {outdir}/figures/blackjack_algorithm_comparison.png")

    print("\n=== Experiment complete ===")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Blackjack RL Experiments (VI/PI/SARSA)")
    p.add_argument("--episodes", type=int, default=20000, help="Episodes for model-free methods")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=str, default="..", help="Output root directory (default: parent of src)")

    args = p.parse_args()
    main(args)
