import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, **kwargs):
        super(CNN1D, self).__init__()

        self.conv_layers = nn.ModuleList()

        current_channels = kwargs["input_channels"]
        current_seq_length = kwargs["seq_length"]

        num_filters = kwargs["num_filters"]
        kernel_sizes = kwargs["kernel_sizes"]

        if len(num_filters) != len(kernel_sizes):
            raise ValueError(
                "The lengths of 'num_filters' and 'kernel_sizes' must match."
            )

        for nf, ks in zip(num_filters, kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(current_channels, nf, kernel_size=ks, padding=ks // 2)
            )

            current_channels = nf
            current_seq_length //= 2

        fc_units = kwargs.get("fc_units", 1024)
        self.fc_layer = nn.Linear(current_seq_length * current_channels, fc_units)

        self.output_layer = nn.Linear(fc_units, 1)

        self._init_weights()

    def _init_weights(self):
        for conv in self.conv_layers:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.fc_layer.weight)
        nn.init.zeros_(self.fc_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = x.reshape(x.size(0), 4, -1)

        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool1d(x, kernel_size=2)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_layer(x))
        x = self.output_layer(x).squeeze(1)
        return x
