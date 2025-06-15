import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle


class SeqActivityDataset(Dataset):
    def __init__(self, data_dict, is_onehot, binary_length=7):
        self.inputs = list(data_dict.keys())
        self.labels = list(data_dict.values())
        self.is_onehot = is_onehot
        self.binary_length = binary_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        dna_sequence = self.inputs[index]
        label = self.labels[index]
        if self.is_onehot:
            input_tensor = torch.tensor(
                self.dna_to_onehot_sequence(dna_sequence), dtype=torch.float32
            )
        else:
            input_tensor = dna_sequence
        label_tensor = label

        return input_tensor, label_tensor

    @staticmethod
    def dna_to_onehot_sequence(dna_sequence):
        mapping = {
            "A": [1, 0, 0, 0],
            "T": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "G": [0, 0, 0, 1],
        }
        onehot_sequence = []
        for base in dna_sequence:
            onehot_sequence.extend(mapping[base])
        return onehot_sequence


def create_dataloader(path, cfg):
    torch.manual_seed(cfg.seed)
    with open(path, "rb") as pkl_file:
        data_dict = pickle.load(pkl_file)
    dataset = SeqActivityDataset(data_dict, True)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
