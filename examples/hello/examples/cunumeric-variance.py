from typing import Any

import cunumeric
import numpy as np
from hello import square, sum, to_scalar

from legate.core import Store


def mean_and_variance(a: Any, n: int) -> float:
    a_sq: Store = square(a)  # A 1-D array of shape (4,)
    sum_sq: Store = sum(a_sq)  # A scalar sum
    sum_a: Store = sum(a)  # A scalar sum

    # Extract scalar values from the Legate stores
    mean_a: float = to_scalar(sum_a) / n
    mean_sum_sq: float = to_scalar(sum_sq) / n
    variance = mean_sum_sq - mean_a * mean_a
    return mean_a, variance


# Example: Use a random array from cunumeric
n = 4
a = cunumeric.random.randn(n).astype(np.float32)
print(a)
print(mean_and_variance(a, n))
