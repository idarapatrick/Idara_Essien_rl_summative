"""
Trains three Policy Gradient algorithms on MicroscopyAMREnv:
  - PPO  (Stable Baselines 3)
  - A2C  (Stable Baselines 3)
  - REINFORCE (custom implementation - not from SB3)

Each algorithm runs a 10-configuration hyperparameter sweep.

Run:
    python training/pg_training.py --algo all      # run all three
    python training/pg_training.py --algo ppo
    python training/pg_training.py --algo a2c
    python training/pg_training.py --algo reinforce
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import MicroscopyAMREnv

TOTAL_TIMESTEPS = 80_000
EVAL_FREQ       = 5_000
N_EVAL_EPISODES = 10

for d in ["results/ppo", "results/a2c", "results/reinforce",
          "models/pg/ppo", "models/pg/a2c", "models/pg/reinforce"]:
    os.makedirs(d, exist_ok=True)


# PPO — 10 hyperparameter configurations

PPO_GRID = [
    # Run 1 — SB3 default-like
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
     "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    # Run 2 — smaller steps
    {"learning_rate": 3e-4, "n_steps": 512,  "batch_size": 64,
     "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    # Run 3 — low LR
    {"learning_rate": 1e-4, "n_steps": 2048, "batch_size": 64,
     "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
    # Run 4 — high clip range
    {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 32,
     "gamma": 0.99, "gae_lambda": 0.90, "clip_range": 0.3, "ent_coef": 0.0},
    # Run 5 — narrow clip range
    {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 64,
     "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.0},
    # Run 6 — entropy for exploration
    {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128,
     "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.05},
    # Run 7 — low gamma
    {"learning_rate": 3e-4, "n_steps": 512,  "batch_size": 32,
     "gamma": 0.90, "gae_lambda": 0.90, "clip_range": 0.2, "ent_coef": 0.0},
    # Run 8 — high gamma + large batch
    {"learning_rate": 1e-4, "n_steps": 2048, "batch_size": 128,
     "gamma": 0.999,"gae_lambda": 0.99, "clip_range": 0.2, "ent_coef": 0.0},
    # Run 9 — high LR aggressive
    {"learning_rate": 1e-3, "n_steps": 512,  "batch_size": 64,
     "gamma": 0.95, "gae_lambda": 0.90, "clip_range": 0.3, "ent_coef": 0.01},
    # Run 10 — balanced tuned
    {"learning_rate": 2e-4, "n_steps": 1024, "batch_size": 64,
     "gamma": 0.97, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.005},
]


# A2C — 10 hyperparameter configurations

A2C_GRID = [
    {"learning_rate": 7e-4, "n_steps": 5,  "gamma": 0.99, "gae_lambda": 1.0,  "ent_coef": 0.0},
    {"learning_rate": 7e-4, "n_steps": 10, "gamma": 0.99, "gae_lambda": 1.0,  "ent_coef": 0.01},
    {"learning_rate": 1e-3, "n_steps": 5,  "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0},
    {"learning_rate": 5e-4, "n_steps": 20, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0},
    {"learning_rate": 7e-4, "n_steps": 5,  "gamma": 0.90, "gae_lambda": 0.95, "ent_coef": 0.0},
    {"learning_rate": 7e-4, "n_steps": 5,  "gamma": 0.99, "gae_lambda": 0.90, "ent_coef": 0.05},
    {"learning_rate": 3e-4, "n_steps": 10, "gamma": 0.99, "gae_lambda": 1.0,  "ent_coef": 0.01},
    {"learning_rate": 1e-3, "n_steps": 20, "gamma": 0.95, "gae_lambda": 0.95, "ent_coef": 0.0},
    {"learning_rate": 5e-4, "n_steps": 5,  "gamma": 0.99, "gae_lambda": 0.99, "ent_coef": 0.1},
    {"learning_rate": 7e-4, "n_steps": 10, "gamma": 0.97, "gae_lambda": 0.95, "ent_coef": 0.005},
]

# REINFORCE — custom implementation

REINFORCE_GRID = [
    {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 5e-4, "gamma": 0.99, "entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 1e-4, "gamma": 0.99, "entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.95, "entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.01, "hidden_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.99, "entropy_coef": 0.05, "hidden_size": 128},
    {"learning_rate": 5e-4, "gamma": 0.99, "entropy_coef": 0.01, "hidden_size": 128},
    {"learning_rate": 2e-3, "gamma": 0.99, "entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 1e-3, "gamma": 0.999,"entropy_coef": 0.0,  "hidden_size": 64},
    {"learning_rate": 5e-4, "gamma": 0.97, "entropy_coef": 0.01, "hidden_size": 128},
]


# Custom REINFORCE policy network and training loop

class PolicyNetwork(nn.Module):
    """Simple MLP policy for REINFORCE."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def select_action(self, obs: np.ndarray):
        x = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(x)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


def compute_returns(rewards: list, gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t for a single episode."""
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    # Normalise for training stability
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train_reinforce(hyperparams: dict, run_id: int) -> dict:
    """Train one REINFORCE configuration."""
    run_name = f"reinforce_run_{run_id:02d}"
    print(f"\n{'='*60}")
    print(f"  REINFORCE {run_name}")
    print(f"  {hyperparams}")
    print(f"{'='*60}")

    env      = MicroscopyAMREnv(seed=run_id)
    eval_env = MicroscopyAMREnv(seed=run_id + 100)

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy    = PolicyNetwork(obs_dim, n_actions, hyperparams["hidden_size"])
    optimizer = optim.Adam(policy.parameters(), lr=hyperparams["learning_rate"])

    gamma       = hyperparams["gamma"]
    ent_coef    = hyperparams["entropy_coef"]
    episode_rewards = []
    total_steps  = 0
    t0 = time.time()

    while total_steps < TOTAL_TIMESTEPS:
        # Collect one episode 
        obs, _ = env.reset()
        log_probs, entropies, rewards = [], [], []
        ep_reward = 0.0

        while True:
            action, log_prob, entropy = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            ep_reward += reward
            total_steps += 1
            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)

        # REINFORCE update
        returns   = compute_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        policy_loss  = -(log_probs * returns).mean()
        entropy_loss = -entropies.mean()
        loss = policy_loss + ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        if len(episode_rewards) % 20 == 0:
            recent = np.mean(episode_rewards[-20:])
            print(f"  Steps={total_steps:>6} | Episodes={len(episode_rewards):>4} | "
                  f"Mean reward (last 20)={recent:.2f}")

    elapsed = time.time() - t0

    # Evaluate
    def eval_policy(n=20):
        ep_rewards = []
        for _ in range(n):
            obs, _ = eval_env.reset()
            total = 0.0
            while True:
                x = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    probs = policy(x)
                action = probs.argmax(dim=-1).item()
                obs, r, term, trunc, _ = eval_env.step(action)
                total += r
                if term or trunc:
                    break
            ep_rewards.append(total)
        return float(np.mean(ep_rewards)), float(np.std(ep_rewards))

    mean_reward, std_reward = eval_policy()

    # Save model
    save_path = f"models/pg/reinforce/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(save_path, "policy.pt"))
    np.save(os.path.join("results/reinforce", f"{run_name}_rewards.npy"),
            np.array(episode_rewards))

    results = {
        "run_id":           run_id,
        "run_name":         run_name,
        "learning_rate":    hyperparams["learning_rate"],
        "gamma":            hyperparams["gamma"],
        "entropy_coef":     hyperparams["entropy_coef"],
        "hidden_size":      hyperparams["hidden_size"],
        "mean_reward":      round(mean_reward, 3),
        "std_reward":       round(std_reward, 3),
        "n_episodes":       len(episode_rewards),
        "training_time_s":  round(elapsed, 1),
        "episode_rewards":  episode_rewards,
    }

    print(f"  Result: mean_reward={mean_reward:.3f} ± {std_reward:.3f}  time={elapsed:.0f}s")
    env.close()
    eval_env.close()
    return results


def _load_completed_results(results_dir: str, algo_name: str) -> list:
    """Load any previously saved per-run result rows from disk."""
    completed = []
    for fname in sorted(os.listdir(results_dir)):
        row_path = os.path.join(results_dir, fname, "run_result.json")
        if os.path.exists(row_path):
            with open(row_path) as f:
                completed.append(json.load(f))
    return completed


def train_sb3_algo(AlgoClass, grid: list, algo_name: str):
    """Generic SB3 training sweep for PPO or A2C.
    Resumes automatically: skips any run whose final_model.zip already exists.
    """
    results_dir = f"results/{algo_name.lower()}"
    model_dir   = f"models/pg/{algo_name.lower()}"
    all_results = _load_completed_results(results_dir, algo_name)
    completed_names = {r["run_name"] for r in all_results}

    for i, hp in enumerate(grid):
        run_name = f"{algo_name.lower()}_run_{i+1:02d}"

        # Resume check
        final_model_path = os.path.join(model_dir, run_name, "final_model.zip")
        if run_name in completed_names or os.path.exists(final_model_path):
            print(f"\n  SKIPPING {run_name} - already completed.")
            continue

        print(f"  {algo_name} {run_name}")
        print(f"  {hp}")


        env      = Monitor(MicroscopyAMREnv(seed=i + 1))
        eval_env = Monitor(MicroscopyAMREnv(seed=i + 101))

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, run_name),
            log_path=os.path.join(results_dir, run_name),
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=0,
        )

        kwargs = dict(
            policy="MlpPolicy",
            env=env,
            learning_rate=hp["learning_rate"],
            gamma=hp["gamma"],
            verbose=0,
            tensorboard_log=f"logs/{algo_name.lower()}/{run_name}",
            seed=i + 1,
        )

        if AlgoClass == PPO:
            kwargs.update({
                "n_steps":    hp["n_steps"],
                "batch_size": hp["batch_size"],
                "gae_lambda": hp["gae_lambda"],
                "clip_range": hp["clip_range"],
                "ent_coef":   hp["ent_coef"],
            })
        elif AlgoClass == A2C:
            kwargs.update({
                "n_steps":    hp["n_steps"],
                "gae_lambda": hp["gae_lambda"],
                "ent_coef":   hp["ent_coef"],
            })

        t0 = time.time()
        model = AlgoClass(**kwargs)

        # Track episode rewards via monitor
        ep_rewards = []

        class EpRewardCB(EvalCallback.__class__):
            pass

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_cb,
            progress_bar=True,
            reset_num_timesteps=True,
        )
        elapsed = time.time() - t0

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )

        model.save(os.path.join(model_dir, run_name, "final_model"))

        row = {k: v for k, v in hp.items()}
        row.update({
            "run_id":          i + 1,
            "run_name":        run_name,
            "mean_reward":     round(mean_reward, 3),
            "std_reward":      round(std_reward, 3),
            "training_time_s": round(elapsed, 1),
        })
        # Save immediately - survives power loss
        run_result_dir = os.path.join(results_dir, run_name)
        os.makedirs(run_result_dir, exist_ok=True)
        with open(os.path.join(run_result_dir, "run_result.json"), "w") as f:
            json.dump(row, f, indent=2)

        all_results.append(row)
        print(f"  Result: mean_reward={mean_reward:.3f} ± {std_reward:.3f}  time={elapsed:.0f}s")

        env.close()
        eval_env.close()

    # Rebuild from disk - picks up any runs completed before a crash
    all_results = _load_completed_results(results_dir, algo_name)
    if not all_results:
        print(f"\n{algo_name}: no completed runs found yet.")
        return []
    summary = pd.DataFrame(all_results).sort_values("mean_reward", ascending=False)
    summary.to_csv(os.path.join(results_dir, f"{algo_name.lower()}_results_summary.csv"), index=False)
    print(f"\n{algo_name} SWEEP COMPLETE")
    print(summary[["run_name","learning_rate","gamma","mean_reward","std_reward"]].to_string())

    best = {"algorithm": algo_name,
            "run_name":  summary.iloc[0]["run_name"],
            "mean_reward": summary.iloc[0]["mean_reward"]}
    with open(os.path.join(model_dir, "best_run.json"), "w") as f:
        json.dump(best, f, indent=2)

    return all_results


def run_reinforce_sweep():
    all_results = []
    for i, hp in enumerate(REINFORCE_GRID):
        run_name   = f"reinforce_run_{i+1:02d}"
        done_path  = os.path.join("results/reinforce", run_name, "run_result.json")
        model_path = os.path.join("models/pg/reinforce", run_name, "policy.pt")
        if os.path.exists(done_path) or os.path.exists(model_path):
            print(f"\n  SKIPPING {run_name} - already completed.")
            if os.path.exists(done_path):
                with open(done_path) as f:
                    all_results.append(json.load(f))
            continue
        result = train_reinforce(hp, run_id=i + 1)
        run_result_dir = os.path.join("results/reinforce", run_name)
        os.makedirs(run_result_dir, exist_ok=True)
        row = {k: v for k, v in result.items() if k != "episode_rewards"}
        with open(os.path.join(run_result_dir, "run_result.json"), "w") as f:
            json.dump(row, f, indent=2)
        all_results.append(result)

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "episode_rewards"}
        for r in all_results
    ]).sort_values("mean_reward", ascending=False)
    summary.to_csv("results/reinforce/reinforce_results_summary.csv", index=False)
    print("\nREINFORCE SWEEP COMPLETE")
    print(summary[["run_name","learning_rate","gamma","entropy_coef","mean_reward"]].to_string())

    best = {"algorithm": "REINFORCE",
            "run_name":  summary.iloc[0]["run_name"],
            "mean_reward": summary.iloc[0]["mean_reward"]}
    with open("models/pg/reinforce/best_run.json", "w") as f:
        json.dump(best, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="all",
                        choices=["all", "ppo", "a2c", "reinforce"],
                        help="Which algorithm to train")
    args = parser.parse_args()

    if args.algo in ("all", "ppo"):
        train_sb3_algo(PPO, PPO_GRID, "PPO")

    if args.algo in ("all", "a2c"):
        train_sb3_algo(A2C, A2C_GRID, "A2C")

    if args.algo in ("all", "reinforce"):
        run_reinforce_sweep()