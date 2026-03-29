"""
Deep Deterministic Policy Gradient (DDPG) baseline for comparison.

Reference: Lillicrap et al., "Continuous control with deep reinforcement
learning", ICLR 2016.

Usage:
    python baselines/ddpg.py --env Pendulum-v1 --steps 2000
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


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


class DDPGAgent:
    """Minimal DDPG implementation."""

    def __init__(self, state_dim, action_dim, action_max=1.0, lr_actor=1e-4,
                 lr_critic=1e-3, gamma=0.99, tau=0.005, hidden=256):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for DDPG")

        self.gamma = gamma
        self.tau = tau
        self.action_max = action_max

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.noise = OUNoise(action_dim)
        self.buffer = []
        self.buffer_cap = 100000

    def select_action(self, state, explore=True):
        s = torch.FloatTensor(state).unsqueeze(0)
        a = self.actor(s).detach().numpy()[0] * self.action_max
        if explore:
            a += self.noise.sample() * self.action_max * 0.1
        return np.clip(a, -self.action_max, self.action_max)

    def store(self, s, a, r, s2, done):
        if len(self.buffer) >= self.buffer_cap:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, done))

    def update(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in idx])
        s, a, r, s2, d = [torch.FloatTensor(np.array(x)) for x in [s, a, r, s2, d]]

        with torch.no_grad():
            a2 = self.actor_target(s2) * self.action_max
            q_target = r + self.gamma * (1 - d) * self.critic_target(torch.cat([s2, a2], 1)).squeeze()

        q_val = self.critic(torch.cat([s, a], 1)).squeeze()
        critic_loss = nn.MSELoss()(q_val, q_target)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        a_pred = self.actor(s) * self.action_max
        actor_loss = -self.critic(torch.cat([s, a_pred], 1)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)


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
    agent = DDPGAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        action_max=float(env.action_space.high[0]),
    )

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0
    for step in range(args.steps):
        action = agent.select_action(obs)
        obs2, reward, term, trunc, _ = env.step(action)
        agent.store(obs, action, reward, obs2, term or trunc)
        agent.update()
        total_reward += reward
        obs = obs2
        if term or trunc:
            agent.noise.reset()
            obs, _ = env.reset()

    print(f"DDPG on {args.env}: total_reward={total_reward:.1f} over {args.steps} steps")
    env.close()


if __name__ == "__main__":
    main()
