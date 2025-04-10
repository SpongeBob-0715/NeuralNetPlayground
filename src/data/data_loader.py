from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

class CIFAR10Loader:
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
    CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    CIFAR_STD = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)

    @staticmethod
    def load_batches(batch_names, data_dir=DEFAULT_DATA_DIR):
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        X_list, y_list = [], []
        for batch_name in tqdm(batch_names, desc="Loading batches"):
            batch_path = data_dir / batch_name
            if not batch_path.exists():
                raise FileNotFoundError(f"CIFAR batch not found: {batch_path}")
            
            with batch_path.open('rb') as f:
                batch = pickle.load(f, encoding='bytes')
            X_list.append(batch[b'data'].astype(np.float32))
            y_list.append(np.array(batch[b'labels'], dtype=np.int64))

        return np.concatenate(X_list), np.concatenate(y_list)

    @staticmethod
    def preprocess(X, y):
        X = X.reshape(-1, 3, 32, 32)
        X = (X / 255.0 - CIFAR10Loader.CIFAR_MEAN) / CIFAR10Loader.CIFAR_STD
        X = X.reshape(-1, 3072)
        return X, y

    @classmethod
    def load_dataset(cls, train_batch_names, test_batch_name="test_batch", data_dir=DEFAULT_DATA_DIR):
        train_X, train_y = cls.load_batches(train_batch_names, data_dir)
        test_X, test_y = cls.load_batches([test_batch_name], data_dir)
        
        train_X, train_y = cls.preprocess(train_X, train_y)
        test_X, test_y = cls.preprocess(test_X, test_y)
        
        print(f"Loaded training set: samples={len(train_X):,}, classes={len(np.unique(train_y))}")
        print(f"Loaded test set: samples={len(test_X):,}, classes={len(np.unique(test_y))}")
        
        return {
            'train_X': train_X, 'train_y': train_y,
            'test_X': test_X, 'test_y': test_y
        }