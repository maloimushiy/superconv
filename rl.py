import json
import os
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


BASE_LR = 6.3e-4


def learning_rate(kind):
    def schedule(remaining):
        progress = np.clip(1 - remaining, 0, 1)
        if kind == "constant":
            return BASE_LR
        if kind == "cosine":
            return BASE_LR / 1000 + (BASE_LR - BASE_LR / 1000) * (1 + np.cos(np.pi * progress)) / 2
        peak = BASE_LR * (3 if kind == "onecycle" else 1)
        return float(np.interp(progress, [0, 0.4, 0.8, 1], [BASE_LR / 10, peak, BASE_LR / 10, BASE_LR / 1000]))
    if kind not in {"constant", "cosine", "onecycle", "onecycle_low"}:
        raise ValueError(kind)
    return schedule


class LoggedDQN(DQN):
    def train(self, gradient_steps, batch_size=100):
        sampled = self._n_updates % 1000 == 0
        parameters = list(self.q_net.parameters())
        if sampled:
            before = [p.detach().clone() for p in parameters]
        super().train(gradient_steps, batch_size)
        if sampled:
            norm = sum(p.square().sum() for p in before).sqrt()
            change = sum((p.detach() - old).square().sum() for p, old in zip(parameters, before)).sqrt()
            self.diagnostics.append(dict(
                step=self.num_timesteps, updates=self._n_updates,
                td_loss=self.logger.name_to_value["train/loss"],
                lr=self.policy.optimizer.param_groups[0]["lr"],
                epsilon=self.exploration_rate,
                weight_l2=norm.item(),
                grad_l2_clipped=sum(p.grad.square().sum() for p in parameters).sqrt().item(),
                update_ratio=(change / norm.clamp_min(1e-12)).item(),
                update_span=gradient_steps,
            ))


def evaluate(model, env, seeds):
    rewards = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        done, reward_sum = False, 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            reward_sum += reward
            done = terminated or truncated
        rewards.append(reward_sum)
    return np.asarray(rewards)


def record_replay(model, path, seed=30000):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    reward_sum = 0
    try:
        observation, _ = env.reset(seed=seed)
        with imageio.get_writer(path, fps=env.metadata["render_fps"], codec="libx264",
                                macro_block_size=2, ffmpeg_log_level="error") as writer:
            writer.append_data(env.render())
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                reward_sum += reward
                writer.append_data(env.render())
                done = terminated or truncated
    finally:
        env.close()
    return reward_sum


class Evaluation(BaseCallback):
    def __init__(self, folder, interval, episodes, live, live_every, video_every):
        super().__init__()
        self.folder, self.interval, self.episodes = folder, interval, episodes
        self.live, self.live_every, self.video_every = live, live_every, video_every
        self.env = gym.make("LunarLander-v3")
        self.rows, self.training_episodes, self.replays = [], [], []
        self.overhead_seconds = 0
        self.started = time.perf_counter()

    def refresh(self, evaluation=False, video=False):
        started = time.perf_counter()
        train_seconds = started - self.started - self.overhead_seconds
        title = f"{self.folder.parent.name} · seed {self.model.seed} · step {self.num_timesteps}/{self.model._total_timesteps}"
        if evaluation:
            rewards = evaluate(self.model, self.env, range(10000, 10000 + self.episodes))
            self.rows.append(dict(step=self.num_timesteps, reward_mean=rewards.mean(),
                                  reward_std=rewards.std(ddof=1), train_seconds=train_seconds))
            print(f"step={self.num_timesteps}: reward={rewards.mean():.1f} ± {rewards.std(ddof=1):.1f}", flush=True)
        if self.live is not None:
            self.live(self.rows, self.model.diagnostics, title, self.training_episodes)
        if video:
            path = self.folder / "replays" / f"step_{self.num_timesteps:07d}.mp4"
            reward = record_replay(self.model, path)
            self.replays.append(dict(step=self.num_timesteps, seed=30000, reward=reward, file=path.name))
            pd.DataFrame(self.replays).to_csv(self.folder / "replays.csv", index=False)
            if self.live is not None:
                self.live.video(path, f"{title} · replay seed 30000 · reward {reward:.1f}")
        pd.DataFrame(self.rows).to_csv(self.folder / "evaluation.csv", index=False)
        pd.DataFrame(self.model.diagnostics).to_csv(self.folder / "diagnostics.csv", index=False)
        pd.DataFrame(self.training_episodes, columns=["step", "reward", "length"]).to_csv(self.folder / "episodes.csv", index=False)
        self.overhead_seconds += time.perf_counter() - started

    def _on_training_start(self):
        self.started = time.perf_counter()
        self.refresh(evaluation=True, video=bool(self.video_every))

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.training_episodes.append(dict(step=self.num_timesteps, reward=info["episode"]["r"], length=info["episode"]["l"]))
        evaluation = self.num_timesteps % self.interval == 0
        video = bool(self.video_every) and self.num_timesteps % self.video_every == 0
        if self.num_timesteps < self.model._total_timesteps and (evaluation or video or self.num_timesteps % self.live_every == 0):
            self.refresh(evaluation, video)
        return True

    def _on_training_end(self):
        self.refresh(evaluation=True, video=bool(self.video_every))
        self.env.close()


def train_rl(kind="constant", seed=17, steps=100000, output="results/rl",
             eval_every=10000, eval_episodes=10, test_episodes=50,
             live=None, live_every=2000, video_every=25000):
    torch.set_num_threads(1)
    folder = Path(output) / kind / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(schedule=kind, seed=seed, steps=steps, base_lr=BASE_LR,
                  eval_every=eval_every, eval_episodes=eval_episodes, test_episodes=test_episodes,
                  live_every=live_every, video_every=video_every, replay_seed=30000)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    env = gym.make("LunarLander-v3")
    model = LoggedDQN(
        "MlpPolicy", env, learning_rate=learning_rate(kind), seed=seed, device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]), buffer_size=50000, batch_size=128,
        learning_starts=0, gamma=0.99, target_update_interval=250,
        train_freq=4, gradient_steps=-1, exploration_fraction=0.12,
        exploration_final_eps=0.1, verbose=0,
    )
    model.diagnostics = []
    callback = Evaluation(folder, eval_every, eval_episodes, live, live_every, video_every)
    try:
        model.learn(total_timesteps=steps, callback=callback, log_interval=None)
        test_env = gym.make("LunarLander-v3")
        try:
            rewards = evaluate(model, test_env, range(20000, 20000 + test_episodes))
        finally:
            test_env.close()
        pd.DataFrame({"seed": range(20000, 20000 + test_episodes), "reward": rewards}).to_csv(folder / "test_episodes.csv", index=False)
        result = dict(method=kind, seed=seed, steps=model.num_timesteps,
                      reward_mean=rewards.mean(), reward_std=rewards.std(ddof=1),
                      train_seconds=callback.rows[-1]["train_seconds"])
        pd.DataFrame([result]).to_csv(folder / "results.csv", index=False)
        model.save(folder / "model")
        return pd.DataFrame(callback.rows), result
    finally:
        env.close()
        callback.env.close()


if __name__ == "__main__":
    results = [train_rl(kind, seed)[1] for seed in (17, 42, 103)
               for kind in ("constant", "cosine", "onecycle")]
    pd.DataFrame(results).to_csv("results/rl/results.csv", index=False)
