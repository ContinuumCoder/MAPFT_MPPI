"""
Soft Actor-Critic (SAC) baseline for comparison.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor", ICML 2018.

Usage:
    python baselines/sac.py --env Pendulum-v1 --steps 2000
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


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = []
        self.cap = capacity

    def push(self, s, a, r, s2, done):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


class SACAgent:
    """Minimal SAC implementation."""

    def __init__(self, state_dim, action_dim, action_max=1.0, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, hidden=256):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for SAC")

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_max = action_max

        # Networks
        self.actor = self._make_actor(state_dim, action_dim, hidden)
        self.q1 = self._make_q(state_dim, action_dim, hidden)
        self.q2 = self._make_q(state_dim, action_dim, hidden)
        self.q1_target = self._make_q(state_dim, action_dim, hidden)
        self.q2_target = self._make_q(state_dim, action_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        self.buffer = ReplayBuffer()

    def _make_actor(self, s_dim, a_dim, h):
        return nn.Sequential(
            nn.Linear(s_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, a_dim * 2),  # mean + log_std
        )

    def _make_q(self, s_dim, a_dim, h):
        return nn.Sequential(
            nn.Linear(s_dim + a_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0)
        out = self.actor(s)
        mean, log_std = out.chunk(2, dim=-1)
        if deterministic:
            return (torch.tanh(mean) * self.action_max).detach().numpy()[0]
        std = log_std.clamp(-20, 2).exp()
        z = mean + std * torch.randn_like(std)
        return (torch.tanh(z) * self.action_max).detach().numpy()[0]

    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s, a, r, s2, d = [torch.FloatTensor(x) for x in [s, a, r, s2, d]]

        # Q targets
        with torch.no_grad():
            a2_out = self.actor(s2)
            a2_mean, a2_logstd = a2_out.chunk(2, dim=-1)
            a2_std = a2_logstd.clamp(-20, 2).exp()
            a2 = torch.tanh(a2_mean + a2_std * torch.randn_like(a2_std)) * self.action_max
            q1_t = self.q1_target(torch.cat([s2, a2], 1)).squeeze()
            q2_t = self.q2_target(torch.cat([s2, a2], 1)).squeeze()
            target = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)

        q1_val = self.q1(torch.cat([s, a], 1)).squeeze()
        q2_val = self.q2(torch.cat([s, a], 1)).squeeze()
        q1_loss = nn.MSELoss()(q1_val, target)
        q2_loss = nn.MSELoss()(q2_val, target)

        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Actor
        a_out = self.actor(s)
        a_mean, a_logstd = a_out.chunk(2, dim=-1)
        a_std = a_logstd.clamp(-20, 2).exp()
        a_new = torch.tanh(a_mean + a_std * torch.randn_like(a_std)) * self.action_max
        q_new = torch.min(self.q1(torch.cat([s, a_new], 1)),
                          self.q2(torch.cat([s, a_new], 1)))
        actor_loss = -q_new.mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Soft update targets
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
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
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        action_max=float(env.action_space.high[0]),
    )

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0
    for step in range(args.steps):
        action = agent.select_action(obs)
        obs2, reward, term, trunc, _ = env.step(action)
        agent.buffer.push(obs, action, reward, obs2, term or trunc)
        agent.update()
        total_reward += reward
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()

    print(f"SAC on {args.env}: total_reward={total_reward:.1f} over {args.steps} steps")
    env.close()


if __name__ == "__main__":
    main()
