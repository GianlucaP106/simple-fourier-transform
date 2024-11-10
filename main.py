from typing import Any
import numpy as np
from numpy._typing import NDArray
import math

np.random.seed(10)
SAMPLE = np.random.rand(10)


# non-numpy
def dft2(x: NDArray[np.float64]) -> list[float]:
    N = len(x)
    out = []
    for k in range(N):
        xk = 0
        for n in range(N):
            e = np.exp(-1j * 2 * math.pi * k * n * (1 / N))
            xk += e * x[n]
        out.append(xk)
    return out


def dft(x: NDArray[np.float64]) -> NDArray[Any]:
    N = len(x)
    indices = np.arange(N)
    kxn = indices * indices[:, None]
    e = np.exp(-1j * 2 * math.pi * kxn * (1 / N))
    return np.dot(e, x)


if __name__ == "__main__":
    o = dft(SAMPLE)
    print(o)

    o = dft2(SAMPLE)
    print(o)
