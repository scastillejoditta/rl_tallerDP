def extract_grid_slice(V, pass_idx, dest_idx):
    """Extrae V para todas las posiciones (row,col), fijando pass_idx y dest_idx."""
    grid = np.zeros((NUM_ROWS, NUM_COLS))
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            s = encode_state(row, col, pass_idx, dest_idx)
            grid[row, col] = V[s]
    return grid