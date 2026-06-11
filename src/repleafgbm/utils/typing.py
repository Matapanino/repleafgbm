"""Common type aliases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    # Union (not |) because the runtime branch below must stay a plain object.
    ArrayLike = Union[np.ndarray, "pd.DataFrame"]  # noqa: UP007
else:
    ArrayLike = Union[np.ndarray, object]  # noqa: UP007
