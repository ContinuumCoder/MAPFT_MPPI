"""
Proximal Policy Optimization (PPO) baseline for comparison.

Reference: Schulman et al., "Proximal Policy Optimization Algorithms", 2017.

Usage:
    python baselines/ppo.py --env Pendulum-v1 --steps 2000
"""
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class PPOAgent:
    """Minimal PPO (clip) implementation."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, hidden=64):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PPO")

        self.gamma = gamma
        self.clip_eps = clip_eps

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim * 2),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.opt = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.trajectories = []

    def select_action(self, state):
        s = torch.FloatTensor(state).unsqueeze(0)
        out = self.policy(s)
        mean, log_std = out.chunk(2, dim=-1)
        std = log_std.clamp(-2, 1).exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach().numpy()[0], log_prob.item()

    def store(self, s, a, r, log_prob, done):
        self.trajectories.append((s, a, r, log_prob, done))

    def update(self, epochs=4, batch_size=64):
        if len(self.trajectories) < batch_size:
            return
        states, actions, rewards, log_probs, dones = zip(*self.trajectories)
        self.trajectories = []

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(log_probs)

        # Compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(epochs):
            values = self.value(states).squeeze()
            advantages = returns - values.detach()

            out = self.policy(states)
            mean, log_std = out.chunk(2, dim=-1)
            std = log_std.clamp(-2, 1).exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns)
            loss = policy_loss + 0.5 * value_loss

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()), 0.5
            )
            self.opt.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_GYM:
        print("gymnasium required: pip install gymnasium")
        return

    env = gym.make(args.env)
    np.random.seed(args.seed)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0
    for step in range(args.steps):
        action, log_prob = agent.select_action(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs2, reward, term, trunc, _ = env.step(action)
        agent.store(obs, action, reward, log_prob, term or trunc)
        total_reward += reward
        obs = obs2
        if term or trunc:
            agent.update()
            obs, _ = env.reset()

    print(f"PPO on {args.env}: total_reward={total_reward:.1f} over {args.steps} steps")
    env.close()


if __name__ == "__main__":
    main()
