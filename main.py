"""
Entry point: loads the best-performing trained model and runs a live
simulation with the pygame visualizer.

Usage:
    python main.py                      # auto-selects best model across all algos
    python main.py --algo dqn
    python main.py --algo ppo
    python main.py --algo a2c
    python main.py --algo reinforce
    python main.py --episodes 3
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import MicroscopyAMREnv, ACTION_NAMES
from environment.rendering import MicroscopyRenderer


def load_best_model(algo: str):
    """Load the best saved model for the given algorithm."""
    if algo == "dqn":
        from stable_baselines3 import DQN
        best_path = "models/dqn/best_run.json"
        if not os.path.exists(best_path):
            raise FileNotFoundError("No DQN best_run.json found. Run training first.")
        with open(best_path) as f:
            info = json.load(f)
        model_file = os.path.join("models/dqn", info["run_name"], "final_model")
        print(f"Loading DQN: {model_file}  (mean_reward={info['mean_reward']})")
        return DQN.load(model_file), "DQN", info

    elif algo == "ppo":
        from stable_baselines3 import PPO
        best_path = "models/pg/ppo/best_run.json"
        if not os.path.exists(best_path):
            raise FileNotFoundError("No PPO best_run.json found. Run training first.")
        with open(best_path) as f:
            info = json.load(f)
        model_file = os.path.join("models/pg/ppo", info["run_name"], "final_model")
        print(f"Loading PPO: {model_file}  (mean_reward={info['mean_reward']})")
        return PPO.load(model_file), "PPO", info

    elif algo == "a2c":
        from stable_baselines3 import A2C
        best_path = "models/pg/a2c/best_run.json"
        if not os.path.exists(best_path):
            raise FileNotFoundError("No A2C best_run.json found. Run training first.")
        with open(best_path) as f:
            info = json.load(f)
        model_file = os.path.join("models/pg/a2c", info["run_name"], "final_model")
        print(f"Loading A2C: {model_file}  (mean_reward={info['mean_reward']})")
        return A2C.load(model_file), "A2C", info

    elif algo == "reinforce":
        from training.pg_training import PolicyNetwork
        best_path = "models/pg/reinforce/best_run.json"
        if not os.path.exists(best_path):
            raise FileNotFoundError("No REINFORCE best_run.json found. Run training first.")
        with open(best_path) as f:
            info = json.load(f)
        weights = os.path.join("models/pg/reinforce", info["run_name"], "policy.pt")
        obs_dim   = MicroscopyAMREnv().observation_space.shape[0]
        n_actions = MicroscopyAMREnv().action_space.n
        policy = PolicyNetwork(obs_dim, n_actions)
        policy.load_state_dict(torch.load(weights, map_location="cpu"))
        policy.eval()
        print(f"Loading REINFORCE: {weights}  (mean_reward={info['mean_reward']})")
        return policy, "REINFORCE", info

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def auto_select_best_algo():
    """Compare mean_rewards across all trained models and pick the best."""
    candidates = []
    for algo, path in [
        ("dqn",       "models/dqn/best_run.json"),
        ("ppo",       "models/pg/ppo/best_run.json"),
        ("a2c",       "models/pg/a2c/best_run.json"),
        ("reinforce", "models/pg/reinforce/best_run.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
            candidates.append((algo, info["mean_reward"]))

    if not candidates:
        print("No trained models found. Running with random agent for demo.")
        return None

    best_algo = max(candidates, key=lambda x: x[1])
    print(f"\nAuto-selected best algorithm: {best_algo[0].upper()}  "
          f"(mean_reward={best_algo[1]:.3f})")
    return best_algo[0]


def run_episode(model, algo: str, env: MicroscopyAMREnv,
                renderer: MicroscopyRenderer, episode_num: int):
    """Run one episode with the loaded model and live rendering."""
    obs, info = env.reset(seed=episode_num * 7)
    total_reward = 0.0
    step = 0
    last_action = None
    last_reward = None

    print(f"\n--- Episode {episode_num} | "
          f"Antibiotic class: {env.episode_data.class_name} ---")

    while True:
        # Select action
        if algo in ("dqn", "ppo", "a2c"):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif algo == "reinforce":
            x = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = model(x)
            action = int(probs.argmax(dim=-1).item())
        else:  # random fallback
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        last_action = action
        last_reward = reward

        # Render
        renderer.render(
            frame=int(env.current_frame),
            total_frames=env.episode_data.total_frames,
            obs=obs,
            compute_budget=env.compute_budget,
            anomaly_history=env._recent_anomaly_history,
            detection_confidence=env._detection_confidence,
            episode_data=env.episode_data,
            stats={
                "detections":     env._episode_detections,
                "misses":         env._episode_misses,
                "false_alarms":   env._episode_false_alarms,
                "total_reward":   env._total_reward,
                "critical_misses": env._critical_misses,
            },
            last_action=last_action,
            last_reward=last_reward,
        )

        step += 1

        if terminated or truncated:
            break

    print(f"  Steps: {step}  |  Total reward: {total_reward:.2f}  |  "
          f"Detections: {info['episode_detections']}  |  "
          f"Misses: {info['episode_misses']}  |  "
          f"False alarms: {info['episode_false_alarms']}  |  "
          f"Budget remaining: {info['compute_budget']:.1f}%")

    return total_reward


def main():
    parser = argparse.ArgumentParser(
        description="Run best AMR RL agent in live simulation"
    )
    parser.add_argument("--algo", default=None,
                        choices=["dqn", "ppo", "a2c", "reinforce", "random"],
                        help="Algorithm to run (default: auto-select best)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    args = parser.parse_args()

    algo = args.algo or auto_select_best_algo() or "random"

    # Load model
    if algo == "random":
        model = None
        algo_label = "Random"
        best_info = {"mean_reward": "N/A"}
    else:
        model, algo_label, best_info = load_best_model(algo)

    # Create environment and renderer
    env = MicroscopyAMREnv(render_mode="human")
    renderer = MicroscopyRenderer()

    print(f"\nRunning {args.episodes} episode(s) with {algo_label}")
    print("Close the pygame window or press Ctrl+C to quit.\n")

    all_rewards = []
    try:
        for ep in range(1, args.episodes + 1):
            reward = run_episode(model, algo, env, renderer, episode_num=ep)
            all_rewards.append(reward)

        print(f"  {algo_label} — {args.episodes} episodes complete")
        print(f"  Mean reward : {np.mean(all_rewards):.3f}")
        print(f"  Std reward  : {np.std(all_rewards):.3f}")
        print(f"  Best episode: {max(all_rewards):.3f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        renderer.close()


if __name__ == "__main__":
    main()