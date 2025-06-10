import numpy as np
from pathlib import Path
META_DIR = Path('meta')
x_train_meta = np.load(META_DIR / 'x_train_meta.npy')
print(x_train_meta.shape)
print(x_train_meta[:7])