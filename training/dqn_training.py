"""
Trains DQN on the MicroscopyAMREnv using Stable Baselines 3.
Runs a 10-configuration hyperparameter sweep and saves results.

Run:
    python training/dqn_training.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import MicroscopyAMREnv

TOTAL_TIMESTEPS = 80_000
EVAL_FREQ       = 5_000
N_EVAL_EPISODES = 10
RESULTS_DIR     = "results/dqn"
MODEL_DIR       = "models/dqn"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# 10 hyperparameter configurations 
# Vary: learning_rate, buffer_size, gamma, batch_size, target_update_interval
HYPERPARAMETER_GRID = [
    # Run 1 - conservative baseline
    {"learning_rate": 1e-4, "buffer_size": 10_000, "gamma": 0.99,
     "batch_size": 32,  "target_update_interval": 1000, "exploration_fraction": 0.15},
    # Run 2 - higher LR
    {"learning_rate": 5e-4, "buffer_size": 10_000, "gamma": 0.99,
     "batch_size": 32,  "target_update_interval": 1000, "exploration_fraction": 0.15},
    # Run 3 - large buffer
    {"learning_rate": 1e-4, "buffer_size": 50_000, "gamma": 0.99,
     "batch_size": 64,  "target_update_interval": 500,  "exploration_fraction": 0.20},
    # Run 4 - low gamma (myopic)
    {"learning_rate": 1e-4, "buffer_size": 10_000, "gamma": 0.90,
     "batch_size": 32,  "target_update_interval": 1000, "exploration_fraction": 0.15},
    # Run 5 - high gamma (far-sighted)
    {"learning_rate": 1e-4, "buffer_size": 10_000, "gamma": 0.999,
     "batch_size": 64,  "target_update_interval": 500,  "exploration_fraction": 0.10},
    # Run 6 - large batch
    {"learning_rate": 3e-4, "buffer_size": 50_000, "gamma": 0.99,
     "batch_size": 128, "target_update_interval": 500,  "exploration_fraction": 0.20},
    # Run 7 - aggressive exploration
    {"learning_rate": 1e-4, "buffer_size": 20_000, "gamma": 0.95,
     "batch_size": 32,  "target_update_interval": 1000, "exploration_fraction": 0.40},
    # Run 8 - fast target update
    {"learning_rate": 5e-4, "buffer_size": 20_000, "gamma": 0.99,
     "batch_size": 64,  "target_update_interval": 250,  "exploration_fraction": 0.15},
    # Run 9 - high LR + large buffer
    {"learning_rate": 1e-3, "buffer_size": 50_000, "gamma": 0.99,
     "batch_size": 128, "target_update_interval": 1000, "exploration_fraction": 0.20},
    # Run 10 - balanced, tuned
    {"learning_rate": 3e-4, "buffer_size": 30_000, "gamma": 0.97,
     "batch_size": 64,  "target_update_interval": 750,  "exploration_fraction": 0.12},
]


class RewardLoggerCallback(BaseCallback):
    """Logs per-episode rewards for learning curve plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._current_episode_reward += reward
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True


def train_dqn(hyperparams: dict, run_id: int) -> dict:
    """Train one DQN configuration and return results dict."""
    run_name = f"dqn_run_{run_id:02d}"
    print(f"\n{'='*60}")
    print(f"  DQN {run_name}")
    print(f"  {hyperparams}")
    print(f"{'='*60}")

    env      = Monitor(MicroscopyAMREnv(seed=run_id))
    eval_env = Monitor(MicroscopyAMREnv(seed=run_id + 100))

    reward_cb = RewardLoggerCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, run_name),
        log_path=os.path.join(RESULTS_DIR, run_name),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=hyperparams["learning_rate"],
        buffer_size=hyperparams["buffer_size"],
        gamma=hyperparams["gamma"],
        batch_size=hyperparams["batch_size"],
        target_update_interval=hyperparams["target_update_interval"],
        exploration_fraction=hyperparams["exploration_fraction"],
        exploration_final_eps=0.02,
        learning_starts=1000,
        train_freq=4,
        verbose=0,
        tensorboard_log=f"logs/dqn/{run_name}",
        seed=run_id,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[reward_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )

    results = {
        "run_id":                   run_id,
        "run_name":                 run_name,
        "learning_rate":            hyperparams["learning_rate"],
        "buffer_size":              hyperparams["buffer_size"],
        "gamma":                    hyperparams["gamma"],
        "batch_size":               hyperparams["batch_size"],
        "target_update_interval":   hyperparams["target_update_interval"],
        "exploration_fraction":     hyperparams["exploration_fraction"],
        "mean_reward":              round(mean_reward, 3),
        "std_reward":               round(std_reward, 3),
        "n_episodes_trained":       len(reward_cb.episode_rewards),
        "training_time_s":          round(elapsed, 1),
        "episode_rewards":          reward_cb.episode_rewards,
    }

    # Save model
    model.save(os.path.join(MODEL_DIR, run_name, "final_model"))
    # Save episode rewards for learning curves
    np.save(
        os.path.join(RESULTS_DIR, run_name, "episode_rewards.npy"),
        np.array(reward_cb.episode_rewards)
    )

    print(f"  Result: mean_reward={mean_reward:.3f} ± {std_reward:.3f}  "
          f"time={elapsed:.0f}s")

    env.close()
    eval_env.close()
    return results


def run_sweep():
    all_results = []
    for i, hp in enumerate(HYPERPARAMETER_GRID):
        result = train_dqn(hp, run_id=i + 1)
        all_results.append(result)

    # Save summary table
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "episode_rewards"}
        for r in all_results
    ])
    summary = summary.sort_values("mean_reward", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "dqn_results_summary.csv"), index=False)

    print("DQN SWEEP COMPLETE")
    print(summary[["run_name","learning_rate","gamma","batch_size","mean_reward","std_reward"]].to_string())
    print(f"\nBest run: {summary.iloc[0]['run_name']}  "
          f"mean_reward={summary.iloc[0]['mean_reward']}")

    # Save best run name for main.py
    best = {"algorithm": "DQN", "run_name": summary.iloc[0]["run_name"],
            "mean_reward": summary.iloc[0]["mean_reward"]}
    with open(os.path.join(MODEL_DIR, "best_run.json"), "w") as f:
        json.dump(best, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_sweep()
