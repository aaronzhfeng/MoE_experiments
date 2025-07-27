import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

def train_val_split(data_root:str, test_size=0.15):
    data = np.load(os.path.join(data_root, 'all_data.npz'), allow_pickle=True)
    x, y, x_pos, y_pos = data['X'], data['Y'], data['x_pos'], data['y_pos'] # Shape: [num_samples, seq_len]
    x_train, x_val, y_train, y_val, x_pos_train, x_pos_val, y_pos_train, y_pos_val = train_test_split(
        x, y, x_pos, y_pos, test_size=test_size
    )
    print('[Dataset Size]', len(y_train), len(y_val))
    train_path = os.path.join(data_root, 'train.npz')
    val_path = os.path.join(data_root, 'val.npz')
    np.savez_compressed(train_path, x=x_train, y=y_train, x_pos=x_pos_train, y_pos=y_pos_train)
    np.savez_compressed(val_path, x=x_val, y=y_val, x_pos=x_pos_val, y_pos=y_pos_val)

data_root = sys.argv[1]
test_size = float(sys.argv[2])
train_val_split(data_root, test_size)