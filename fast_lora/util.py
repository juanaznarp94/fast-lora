import numpy as np


def _merge_or_replace(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Tries to merge an old array with a new array, where the new array contains values other than np.nan.
    If the new array has a different shape than the old array, then it is completely replaced.
    """
    if not np.isnan(new).any():
        return new
    elif old.shape == new.shape:
        valid_indices = ~np.isnan(new)
        old[valid_indices] = new[valid_indices]
        return old
    else:
        raise ValueError(
            "Cannot merge arrays of different shapes if there are NaN values in the new array"
        )
