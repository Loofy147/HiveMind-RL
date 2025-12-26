import numpy as np
import random
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
import sys

# ==========================================
# 0. CONFIGURATION
# ==========================================
class Config:
    # Environment settings
    NUM_AGENTS = 20
    GRID_SIZE = 40
    BOUNDS = 40.0
    COMM_RANGE = 15.0
    FRICTION = 0.98
    GRAVITY_STR = 0.1
    BASE_RADIUS = 12.0
    SCOUT_PROPORTION = 0.25
    NUM_WALLS = 3
    NUM_HAZARDS = 10
    NUM_ASTEROIDS = 50

    # Agent settings
    SCOUT_THRUST = 1.5
    MINER_THRUST = 0.8
    SCOUT_ENERGY = 2.5
    MINER_ENERGY = 2.0
    ENERGY_DECAY = 0.0005
    THRUST_ENERGY_COST = 0.005

    # Reward settings
    REWARD_ASTEROID = 20.0
    REWARD_BASE = 100.0
    PENALTY_HAZARD = -5.0
    PENALTY_DEATH = -20.0
    TIME_STEP_PENALTY = -0.01

    # Neural network settings
    HIDDEN_DIM = 128
    N_ACTIONS = 5

    # Training settings
    LEARNING_RATE = 0.0001
    RHO_ALPHA = 0.01
    GAMMA = 0.99
    EPSILON_START = 0.5
    EPSILON_DECAY = 0.9
    EPSILON_MIN = 0.05
    GENERATIONS = 30
    TRAINING_STEPS = 250
    TARGET_UPDATE_FREQ = 5
    CLIP_GRAD_NORM = 1.0
    RHO_CLAMP = (-50.0, 50.0)

    # Playback settings
    PLAYBACK_STEPS = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing LEGION GOLDEN EDITION on: {device}")

if not os.path.exists('outputs'):
    os.makedirs('outputs')

class PopArtScaler:
    """Dynamically normalizes rewards to Mean=0, Std=1."""
    def __init__(self, clip_range=(-5.0, 5.0)):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.clip_range = clip_range

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if isinstance(x, (list, np.ndarray)) else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        normalized_x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normalized_x, self.clip_range[0], self.clip_range[1])

# ==========================================
# 1. SIMULATION ENVIRONMENT
# ==========================================
class LegionEnv:
    def __init__(self, num_agents=Config.NUM_AGENTS, grid_size=Config.GRID_SIZE, randomize=True):
        self.num_agents = num_agents
        self.bounds = Config.BOUNDS
        self.grid_size = grid_size
        self.center = np.array([self.bounds / 2, self.bounds / 2])
        self.comm_range = Config.COMM_RANGE
        self.randomize = randomize

        self.friction = Config.FRICTION
        self.gravity_str = Config.GRAVITY_STR
        self.base_radius = Config.BASE_RADIUS
        self.base_angle = 0.0

        # Classes: 25% Scouts (0), 75% Miners (1)
        n_scouts = int(num_agents * Config.SCOUT_PROPORTION)
        self.classes = np.array([0]*n_scouts + [1]*(num_agents-n_scouts))

        self.reset()

    def _calculate_base_pos(self):
        x = self.center[0] + self.base_radius * math.cos(self.base_angle)
        y = self.center[1] + self.base_radius * math.sin(self.base_angle)
        return np.array([x, y])

    def reset(self):
        self.positions = np.random.rand(self.num_agents, 2) * self.bounds
        self.velocities = np.zeros((self.num_agents, 2))
        # High Energy Start
        self.energy = np.where(self.classes == 0, Config.SCOUT_ENERGY, Config.MINER_ENERGY)
        self.carrying = np.zeros(self.num_agents, dtype=bool)

        if self.randomize:
            self.walls = [
                {'x': np.random.uniform(5, 30), 'y': np.random.uniform(5, 30),
                 'w': np.random.uniform(2, 10), 'h': np.random.uniform(2, 10)}
                for _ in range(Config.NUM_WALLS)
            ]
        else:
            self.walls = [{'x': 15, 'y': 25, 'w': 10, 'h': 2}, {'x': 25, 'y': 10, 'w': 2, 'h': 10}]

        self.hazards = [{'pos': np.random.rand(2)*self.bounds, 'vel': (np.random.rand(2)-0.5)} for _ in range(Config.NUM_HAZARDS)]
        self.asteroids = [{'pos': np.random.rand(2)*self.bounds} for _ in range(Config.NUM_ASTEROIDS)] # Abundance
        self.base_station = self._calculate_base_pos()

        return self.get_state_package()

    def get_adjacency_matrix(self):
        pos = torch.tensor(self.positions, device=device).float()
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dists = torch.norm(diff, dim=2)
        adj = (dists < self.comm_range).float()
        return adj

    def get_state_package(self):
        scale = self.grid_size / self.bounds
        global_grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 1: Danger
        for w in self.walls:
            x1, x2 = int(w['x']*scale), int((w['x']+w['w'])*scale)
            y1, y2 = int(w['y']*scale), int((w['y']+w['h'])*scale)
            x1, x2 = max(0, x1), min(self.grid_size, x2)
            y1, y2 = max(0, y1), min(self.grid_size, y2)
            global_grid[1, y1:y2, x1:x2] = 1.0
        for h in self.hazards:
            px, py = int(h['pos'][0]*scale), int(h['pos'][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: global_grid[1, py, px] = 1.0

        # Channel 2: Reward
        bx, by = int(self.base_station[0]*scale), int(self.base_station[1]*scale)
        if 0<=bx<self.grid_size and 0<=by<self.grid_size: global_grid[2, by, bx] = 1.0
        for a in self.asteroids:
            px, py = int(a['pos'][0]*scale), int(a['pos'][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: global_grid[2, py, px] = 0.5

        # Channel 0: Self
        observations = []
        for i in range(self.num_agents):
            local = global_grid.copy()
            val = 1.0 if self.classes[i] == 0 else 0.5
            px, py = int(self.positions[i][0]*scale), int(self.positions[i][1]*scale)
            if 0<=px<self.grid_size and 0<=py<self.grid_size: local[0, py, px] = val
            observations.append(local)

        return torch.FloatTensor(np.array(observations)).to(device), self.get_adjacency_matrix(), torch.LongTensor(self.classes).to(device)

    def step(self, actions):
        self.base_angle += 0.02
        self.base_station = self._calculate_base_pos()

        thrust_mult = np.where(self.classes == 0, Config.SCOUT_THRUST, Config.MINER_THRUST)
        self.velocities += actions * thrust_mult[:, np.newaxis]

        for i in range(self.num_agents):
            to_center = self.center - self.positions[i]
            dist = np.linalg.norm(to_center)
            if dist > 1.0:
                self.velocities[i] += to_center / dist * (self.gravity_str / (dist/5.0))

        self.velocities *= self.friction
        new_pos = self.positions + self.velocities

        for i in range(self.num_agents):
            px, py = new_pos[i]
            hit = False
            for w in self.walls:
                if w['x'] < px < w['x']+w['w'] and w['y'] < py < w['y']+w['h']: hit = True
            if hit: self.velocities[i] *= -0.5
            else: self.positions[i] = new_pos[i]

        self.positions = np.clip(self.positions, 0, self.bounds)

        for h in self.hazards:
            h['pos'] += h['vel']
            for k in range(2):
                if h['pos'][k] < 0 or h['pos'][k] > self.bounds: h['vel'][k] *= -1

        thrust_mag = np.linalg.norm(actions, axis=1)
        self.energy -= (Config.THRUST_ENERGY_COST * thrust_mag) + Config.ENERGY_DECAY

        rewards = np.full(self.num_agents, Config.TIME_STEP_PENALTY)
        minerals = 0

        # --- REWARD AND PENALTY LOGIC ---
        for a in self.asteroids:
            d = np.linalg.norm(self.positions - a['pos'], axis=1)
            mask = (d < 1.5) & (~self.carrying) & (self.classes == 1)
            if np.any(mask):
                winner = np.where(mask)[0][0]
                self.carrying[winner] = True
                a['pos'] = np.random.rand(2)*self.bounds
                rewards[winner] += Config.REWARD_ASTEROID # High Reward

        d_base = np.linalg.norm(self.positions - self.base_station, axis=1)
        mask_dep = (d_base < 2.0) & (self.carrying)
        if np.any(mask_dep):
            self.carrying[mask_dep] = False
            self.energy[mask_dep] = Config.MINER_ENERGY
            rewards[mask_dep] += Config.REWARD_BASE # Massive Reward
            minerals += np.sum(mask_dep)

        for h in self.hazards:
            d = np.linalg.norm(self.positions - h['pos'], axis=1)
            mask_hit = d < 1.0
            if np.any(mask_hit):
                rewards[mask_hit] += Config.PENALTY_HAZARD
                self.energy[mask_hit] -= 0.5

        dead = self.energy <= 0
        if np.any(dead):
            self.positions[dead] = self.base_station.copy()
            self.velocities[dead] = 0
            self.energy[dead] = 1.0
            self.carrying[dead] = False
            rewards[dead] += Config.PENALTY_DEATH # Reduced Penalty

        return self.get_state_package(), rewards, minerals

# ==========================================
# 2. NEURAL NETWORK (CNN + GNN + GRU)
# ==========================================
class SwarmAttentionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embeddings, adjacency):
        Q = self.W_q(embeddings)
        K = self.W_k(embeddings)
        V = self.W_v(embeddings)
        scores = torch.matmul(Q, K.t()) / math.sqrt(self.embed_dim)
        mask = (adjacency == 0)
        scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        return self.out_proj(context)

class UltimateLegionNet(nn.Module):
    def __init__(self, c_in=3, n_actions=Config.N_ACTIONS, grid_size=Config.GRID_SIZE, hidden_dim=Config.HIDDEN_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        convh = conv2d_size_out(conv2d_size_out(grid_size, 5, 2), 3, 2)
        flat_size = convw * convh * 32

        self.fc_vis = nn.Linear(flat_size, hidden_dim)
        self.cls_emb = nn.Embedding(2, 16)
        self.gru = nn.GRUCell(hidden_dim + 16, hidden_dim)
        self.att = SwarmAttentionHead(hidden_dim)
        self.fc_q = nn.Linear(hidden_dim*2, n_actions)
        self.register_buffer('rho', torch.tensor(0.0))
        self.hidden_dim = hidden_dim

    def forward(self, img, adj, cls, hid):
        x = F.relu(self.conv2(F.relu(self.conv1(img)))).view(img.size(0), -1)
        feat = torch.cat([F.relu(self.fc_vis(x)), self.cls_emb(cls)], dim=1)
        new_hid = self.gru(feat, hid)
        ctx = self.att(new_hid, adj)
        return self.fc_q(torch.cat([new_hid, ctx], dim=1)), new_hid

    def init_hidden(self, bs): return torch.zeros(bs, self.hidden_dim).to(device)

# ==========================================
# 3. META-TRAINER
# ==========================================
class HiveMind:
    def __init__(self):
        self.pop = [{"params": {"lr": Config.LEARNING_RATE, "rho_alpha": Config.RHO_ALPHA}, "score": 0, "model": None} for _ in range(2)]
        self.scaler = PopArtScaler()
    def get_model(self, p):
        pol = UltimateLegionNet().to(device)
        tgt = copy.deepcopy(pol)
        opt = optim.Adam(pol.parameters(), lr=p['lr'])
        return pol, tgt, opt
    def evolve(self):
        self.pop.sort(key=lambda x: x['score'], reverse=True)
        self.pop[-1] = copy.deepcopy(self.pop[0])
        self.pop[-1]['params']['lr'] *= random.uniform(0.9, 1.1)

def train(args):
    print(f"\n>>> TRAINING LEGION (GOLDEN EDITION) | Agents: {args.agents} | Gen: {args.generations}")
    env = LegionEnv(args.agents)
    hive = HiveMind()
    act_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    print(f"{'Gen':<4} | {'Minerals':<8} | {'Surv%':<6} | {'Rho':<8}")
    print("-" * 40)

    for gen in range(args.generations):
        for strat in hive.pop:
            pol, tgt, opt = hive.get_model(strat['params'])
            if strat['model']: pol.load_state_dict(strat['model'])

            (vis, adj, cls) = env.reset()
            hid = pol.init_hidden(args.agents)
            mins = 0

            for t in range(Config.TRAINING_STEPS):
                with torch.no_grad():
                    q, hid = pol(vis, adj, cls, hid)
                    epsilon = max(Config.EPSILON_MIN, Config.EPSILON_START * Config.EPSILON_DECAY**gen)
                    idx = [q[i].argmax().item() if random.random() > epsilon else random.randint(0,4) for i in range(args.agents)]

                (n_vis, n_adj, n_cls), r, m = env.step(np.array([act_map[i] for i in idx]))
                mins += m
                hive.scaler.update(r)
                r_norm = hive.scaler.normalize(r)

                opt.zero_grad()
                hid_d = hid.detach()
                q_act = pol(vis, adj, cls, hid_d)[0][range(args.agents), idx]

                with torch.no_grad():
                    q_next = tgt(n_vis, n_adj, n_cls, hid_d)[0].max(1)[0]

                td = torch.FloatTensor(r_norm).to(device) - pol.rho + Config.GAMMA*q_next
                F.mse_loss(q_act, td).backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), Config.CLIP_GRAD_NORM)
                opt.step()

                # --- ADAPTIVE RHO UPDATE ---
                with torch.no_grad():
                    mag = abs(pol.rho.item())
                    alpha = strat['params']['rho_alpha'] * np.exp(-0.1 * mag)
                    diff = (torch.FloatTensor(r_norm).to(device) + Config.GAMMA*q_next - q_act).mean()
                    pol.rho += alpha * diff
                    pol.rho.clamp_(*Config.RHO_CLAMP) # The Stimulus Clamp

                vis, adj, cls = n_vis, n_adj, n_cls
                if t % Config.TARGET_UPDATE_FREQ == 0: tgt.load_state_dict(pol.state_dict())

            strat['score'] = mins * 1000 + np.mean(env.energy) * 100
            strat['stats'] = (mins, np.mean(env.energy), pol.rho.item())
            strat['model'] = copy.deepcopy(pol.state_dict())

        best = max(hive.pop, key=lambda x: x['score'])
        print(f"{gen+1:<4} | {best['stats'][0]:<8.1f} | {best['stats'][1]:<6.2f} | {best['stats'][2]:<8.3f}")
        hive.evolve()

    torch.save(best['model'], args.save_path)
    print(f"Legion Brain Saved: {args.save_path}")

def play(args):
    print(">>> RENDERING GOLDEN LEGION REPLAY...")
    env = LegionEnv(args.agents, randomize=True)
    model = UltimateLegionNet().to(device)

    if not os.path.exists(args.load_file):
        print("Error: Brain file not found. Train first!")
        return

    model.load_state_dict(torch.load(args.load_file, map_location=device))
    act_map = [np.array([0,0]), np.array([0,1]), np.array([0,-1]), np.array([-1,0]), np.array([1,0])]

    viz_pos = []
    (vis, adj, cls) = env.reset()
    hid = model.init_hidden(args.agents)

    for _ in range(Config.PLAYBACK_STEPS):
        viz_pos.append(env.positions.copy())
        with torch.no_grad():
            q, hid = model(vis, adj, cls, hid)
            act = [act_map[i] for i in q.argmax(1).cpu().numpy()]
        (vis, adj, cls), _, _ = env.step(np.array(act))

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('black')
    ax.set_xlim(0,Config.BOUNDS)
    ax.set_ylim(0,Config.BOUNDS)
    ax.set_title("The Golden Legion: High Reward Swarm")

    stars_x = [a['pos'][0] for a in env.asteroids]
    stars_y = [a['pos'][1] for a in env.asteroids]
    ax.plot(stars_x, stars_y, 'y*', markersize=4, alpha=0.3)

    scat = ax.scatter(viz_pos[0][:, 0], viz_pos[0][:, 1], c=['cyan' if c==0 else 'orange' for c in env.classes], s=50)

    def up(i):
        scat.set_offsets(viz_pos[i])
        return scat,

    ani = animation.FuncAnimation(fig, up, frames=len(viz_pos))
    try:
        out = args.load_file.replace('.pth', '.gif')
        ani.save(out, writer='pillow', fps=30)
        print(f"GIF Saved: {out}")
    except:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'])
    parser.add_argument('--agents', type=int, default=Config.NUM_AGENTS)
    parser.add_argument('--generations', type=int, default=Config.GENERATIONS)
    parser.add_argument('--save_path', type=str, default='outputs/legion_brain_golden.pth')
    parser.add_argument('--load_file', type=str, default='outputs/legion_brain_golden.pth')

    args, _ = parser.parse_known_args()
    if args.mode == 'train': train(args)
    else: play(args)
