import numpy as np

def get_subsample(data, nx=100):
    """Randomly sample 1/nxth entries of a data set."""
    n = len(data)//nx
    print(f"subsampling {n} random particles...")
    idx = np.random.choice(len(data), size=n, replace=False)  # get random indices
    return data[idx]