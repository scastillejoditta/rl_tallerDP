import numpy as np
import gymnasium as gym
from gymnasium import spaces

NUM_ROWS, NUM_COLS = 5, 5
HORIZON = 100

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = 0, 1, 2, 3, 4, 5

INTERNAL_WALLS = [
    ((0, 1), (0, 2)), ((1, 1), (1, 2)),
    ((3, 0), (3, 1)), ((4, 0), (4, 1)),
    ((3, 2), (3, 3)), ((4, 2), (4, 3))
]

def check_wall(row, col, new_row, new_col):
    return ((row, col), (new_row, new_col)) in INTERNAL_WALLS or \
        ((new_row, new_col), (row, col)) in INTERNAL_WALLS

LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

PASS_IN_TAXI = 4
MOVEMENT_ACTIONS = (SOUTH, NORTH, EAST, WEST)
DEFAULT_PROB_RESBALE = 0.2

class MilanTaxiEnv(gym.Env):
    def __init__(self, prob_resbale=DEFAULT_PROB_RESBALE):
        super().__init__()
        self.prob_resbale = float(prob_resbale)
        self.action_space = spaces.Discrete(6)  # SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF
        self.observation_space = spaces.MultiDiscrete([NUM_ROWS, NUM_COLS, 5, 4])
        self.nS = NUM_ROWS * NUM_COLS * 5 * 4   # 500
        self.nA = 6  # 6
        self.state = None
        self.time_step = 0
        self.lastaction = None
        self._delivered_passengers = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._delivered_passengers = 0
        self.time_step = 0
        self.lastaction = None

        # Your code goes here: -------------------------------------
        taxi_row = np.random.randint(NUM_ROWS)
        taxi_col = np.random.randint(NUM_COLS)
        pass_idx, dest_idx = self._spawn_new_passenger()
        # ----------------------------------------------------------

        self.state = (taxi_row, taxi_col, pass_idx, dest_idx)

        return self.state, {}

    def step(self, action):
        row, col, pass_idx, dest_idx = self.state

        new_row, new_col, new_pass_idx, new_dest_idx, reward, dest_reached = \
            self._transitions(row, col, pass_idx, dest_idx, action)

        if action in MOVEMENT_ACTIONS and np.random.random() < self.prob_resbale:
            new_row, new_col = row, col

        self.lastaction = action

        self.time_step += 1
        truncated = self.time_step >= HORIZON
        
        if dest_reached:
            self._delivered_passengers += 1
            if not truncated:
                new_pass_idx, new_dest_idx = self._spawn_new_passenger()

        self.state = (new_row, new_col, new_pass_idx, new_dest_idx)

        return self.state, reward, False, truncated, {"delivered_passengers": self._delivered_passengers}

    def _spawn_new_passenger(self):
        pickup, dropoff = np.random.choice(4, size=2, replace=False)
        return int(pickup), int(dropoff)

    def _transitions(self, row, col, pass_idx, dest_idx, action):
        new_row, new_col = row, col # Si no las cambio, se queda en el mismo punto
        new_pass_idx = pass_idx
        dest_reached = False
        reward = -1

        # Your code goes here: -------------------------------------
        if action == NORTH:
            cand_row = max(row - 1, 0)
            new_row = cand_row
        elif action == SOUTH:
            cand_row = min(row + 1, NUM_ROWS - 1)
            new_row = cand_row
        elif action == EAST:
            cand_col = min(col + 1, NUM_COLS - 1)
            if not check_wall(row, col, row, cand_col):
                new_col = cand_col
        elif action == WEST:
            cand_col = max(col - 1, 0)
            if not check_wall(row, col, row, cand_col):
                new_col = cand_col
        elif action == PICKUP:
            if pass_idx != PASS_IN_TAXI and (row, col) == LOCS[pass_idx]: #Pasajero en calle y taxi en celda de pasajero 
                new_pass_idx = PASS_IN_TAXI
            else:
                reward = -10
        elif action == DROPOFF:
            if pass_idx == PASS_IN_TAXI and (row, col) == LOCS[dest_idx]: #Pasajero en taxi y taxi en celda de destino
                dest_reached = True
                reward = 20
            else:
                reward = -10
        # ----------------------------------------------------------

        return new_row, new_col, new_pass_idx, dest_idx, reward, dest_reached

    def render(self):
        pass

    def close(self):
        pass