import pickle

import numba as nb
import numpy as np
import torch
from torch.utils.data import Dataset


@nb.jit(nopython=True)
def spec_augment(mel_spec: np.ndarray, n_time_masks: int, time_mask_width: int, n_freq_masks: int, freq_mask_width: int):
    offset, begin = 0, 0

    for _ in range(n_time_masks):
        offset = np.random.randint(0, time_mask_width)
        begin = np.random.randint(0, mel_spec.shape[1] - offset)
        mel_spec[:, begin: begin + offset] = 0.0

    for _ in range(n_freq_masks):
        offset = np.random.randint(0, freq_mask_width)
        begin = np.random.randint(0, mel_spec.shape[0] - offset)
        mel_spec[begin: begin + offset, :] = 0.0

    return mel_spec


class PrecomputedSpeechDataset(Dataset):
    """
        API ~ GoogleSpeechDataset, use when training to ensure real-time spec_aug
        __getitem__ -> (x, label) if label else x
    """

    def __init__(self, pkl_path, aug_settings: dict = None, label_map: dict = None, train=False):
        super().__init__()
        self.dataset = load_dataset(pkl_path)   # list of (x, label) or [x]
        self.aug_settings = aug_settings
        self.train = train

        # labels: same as GoogleSpeechDataset
        if label_map is not None:
            self.label_list = []
            #label_2_idx = {v: int(k) for k, v in label_map.items()}
            # The label map is not needed here anymore since the labels in the pickle file
            # are already integer indices.
            for sample in self.dataset:
                if isinstance(sample, tuple) and len(sample) == 2:
                    _, y = sample
                    self.label_list.append(y) # Use the integer label directly
                else:
                    self.label_list.append(None)
        else:
            self.label_list = None


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if isinstance(sample, tuple) and len(sample) == 2:
            x, y = sample
        else:
            x, y = sample, None

        # augment
        if self.train and self.aug_settings is not None:
            if "spec_aug" in self.aug_settings:
                x = spec_augment(x, **self.aug_settings["spec_aug"])

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (n_MFCC, T)

        if self.label_list is not None:
            label = self.label_list[idx]
            label = torch.tensor(label, dtype=torch.long)
            return x, label
        elif y is not None:
            y = torch.tensor(y, dtype=torch.long)
            return x, y
        else:
            return x


def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    print(f"Dataset loaded from {file_path}.")
    return dataset
