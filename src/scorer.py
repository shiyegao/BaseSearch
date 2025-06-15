import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class MultiplierPredictor_onehot(nn.Module):
    def __init__(self, input_size):
        super(MultiplierPredictor_onehot, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc5 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc6 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.5)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)
        nn.init.zeros_(self.fc5.bias)
        nn.init.zeros_(self.fc6.bias)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout(torch.relu(self.bn4(self.fc4(x))))
        x = self.dropout(torch.relu(self.bn5(self.fc5(x))))
        x = self.fc6(x).squeeze(1)
        return x


class ConvNet1D(nn.Module):
    def __init__(self, input_channels, seq_length, num_filters, kernel_sizes):
        super(ConvNet1D, self).__init__()

        self.conv_layers = nn.ModuleList()

        current_channels = input_channels
        current_seq_length = seq_length

        num_filters = num_filters
        kernel_sizes = kernel_sizes

        if len(num_filters) != len(kernel_sizes):
            raise ValueError(
                "The lengths of 'num_filters' and 'kernel_sizes' must match."
            )

        # 动态构建卷积层
        for nf, ks in zip(num_filters, kernel_sizes):
            # 添加卷积层
            self.conv_layers.append(
                nn.Conv1d(current_channels, nf, kernel_size=ks, padding=ks // 2)
            )

            # 更新通道数和序列长度
            current_channels = nf
            current_seq_length //= 2  # 假设每层都有池化操作

        # 全连接层
        fc_units = 1024
        self.fc_layer = nn.Linear(current_seq_length * current_channels, fc_units)

        # 输出层
        self.output_layer = nn.Linear(fc_units, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化卷积层的权重
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        # 初始化全连接层的权重
        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = x.reshape(x.size(0), 4, -1)
        # 遍历卷积层
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool1d(x, kernel_size=2)  # 池化操作

        # 展平为全连接层输入
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc_layer(x))

        # 输出层
        x = self.output_layer(x).squeeze(1)
        return x


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


class Predictor:
    NAME = None

    def __init__(self, root_dir):
        self.root_dir = root_dir
        with open(f"{self.root_dir}/{self.NAME}.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.cnt = 0

    @torch.no_grad()
    def predict(self, input_data):
        input_data = [dna_to_onehot_sequence(input_data)]
        predictions = self.model.predict(input_data)
        self.cnt += 1
        return predictions[0]


class CudaPredictor:
    NAME = None

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.cnt = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def predict(self, input_data):
        input_data = (
            torch.tensor(dna_to_onehot_sequence(input_data), dtype=torch.float32)
            .to(self.device)
            .reshape(1, -1)
        )
        predictions = self.model(input_data)
        self.cnt += 1
        return predictions.item()


class MLPPredictor(CudaPredictor):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.model = MultiplierPredictor_onehot(input_size=240).to(self.device)
        self.model.load_state_dict(torch.load(f"{self.root_dir}/{self.NAME}.pth"))
        self.model.eval()


class MaizeRFPredictor(Predictor):
    NAME = "maize_rf"


class MaizeAdaboostPredictor(Predictor):
    NAME = "maize_adaboost"


class MaizeConvNetPredictor(CudaPredictor):
    NAME = "maize_ConvNet"

    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.model = ConvNet1D(
            input_channels=4, seq_length=60, num_filters=[128, 128], kernel_sizes=[5, 5]
        ).to(self.device)
        self.model.load_state_dict(torch.load(f"{self.root_dir}/{self.NAME}.pth"))
        self.model.eval()


class MultiScorer:
    def __init__(self, scorers: List[str], root_dir: str):
        self.scorers = [globals()[scorer](root_dir) for scorer in scorers]
        self.cnt = 0

    @torch.no_grad()
    def predict(self, input_data):
        _sum = sum([scorer.predict(input_data) for scorer in self.scorers])
        self.cnt += 1
        return _sum / len(self.scorers)


def load_scorer(scorer_name: Union[str, List[str]], root_dir: str):
    if isinstance(scorer_name, str):
        return globals()[scorer_name](root_dir)
    else:
        return MultiScorer(scorer_name, root_dir)
