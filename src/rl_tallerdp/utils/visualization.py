import numpy as np
from rl_tallerdp.envs.milan_taxi import NUM_COLS, NUM_ROWS
from rl_tallerdp.models.build_mdp import encode_state


def extract_grid_slice(V, pass_idx, dest_idx):
    """Extrae V para todas las posiciones (row, col), fijando pass_idx y dest_idx."""
    grid = np.zeros((NUM_ROWS, NUM_COLS))
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            s = encode_state(row, col, pass_idx, dest_idx)
            grid[row, col] = V[s]
    return grid


def plot_vi_convergence(deltas_by_label, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, deltas in deltas_by_label.items():
        ax.plot(range(1, len(deltas) + 1), deltas, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("sweep de value iteration")
    ax.set_ylabel(r"$\|V_{k+1} - V_k\|_\infty$")
    ax.set_title("Convergencia de Value Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
