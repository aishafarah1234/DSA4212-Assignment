import numpy as np

def two_basin(x, alpha=10, distance=4, delta=1):
    """
    Two-basin multimodal objective function.

    Global minimum:
        f(b) = -delta

    Parameters:
        x: Candidate point, e.g. [x1, x2]
        alpha: Controls global basin narrowness
        distance: Separation between basin centres
        delta: Depth advantage of global minimum
    """

    x = np.asarray(x, dtype=float)

    # Local basin centre
    a = np.array([-2.0, 0.0])

    # Global basin centre
    b = a + np.array([distance, 0.0])

    # Broad local basin
    local = np.sum((x - a)**2)

    # Narrow global basin
    global_basin = alpha * np.sum((x - b)**2) - delta

    return float(min(local, global_basin))