import numpy as np
import cupy as cp # type: ignore

def _to(data:np.ndarray|cp.ndarray, xp) -> np.ndarray | cp.ndarray:
    if data.device == 'cpu':
        if xp is cp:
            return xp.asarray(data)
        else:
            return data
    else:
        if xp is cp:
            return data
        else:
            return cp.asnumpy(data)