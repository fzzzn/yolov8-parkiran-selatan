import numpy as np

# 🟧 ORANGE AREA (upper parking section)
ORANGE_AREA = np.array([
    (1758,  172),    # top-left
    (2557,  211),    # top-right
    (2558,  561),    # bottom-right
    (1527,  413),    # bottom-left
], dtype=np.int32)

# 🟥 RED AREA (lower parking section)
RED_AREA = np.array([
    (1535,  417),    # top-left
    (2557,  567),    # top-right
    (2553, 1432),    # bottom-right
    (1035, 1442),    # bottom-left
], dtype=np.int32)
