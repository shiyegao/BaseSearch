project_name = "ste-as-maize"
name = "ste-as-maize"
species = "Maize"
model_name = [
    # {"mlp_onehot": {"input_size": 240, "hidden_layer_sizes": [1024, 1024, 1024, 1024, 1024]}},
    {
        "ConvNet": {
            "input_channels": 4,
            "seq_length": 60,
            "num_filters": [128, 128],
            "kernel_sizes": [5, 5],
            "fc_units": 1024,
        }
    },
    {
        "ConvNet": {
            "input_channels": 4,
            "seq_length": 60,
            "num_filters": [128],
            "kernel_sizes": [5],
            "fc_units": 1024,
        }
    },
    {
        "ConvNet": {
            "input_channels": 4,
            "seq_length": 60,
            "num_filters": [128, 128, 128],
            "kernel_sizes": [5, 5, 5],
            "fc_units": 1024,
        }
    },
]

batch_size = [4, 8, 16, 32, 64, 128]
learning_rate = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
weight_decay = [1e-6, 1e-5, 1e-4, 1e-3]
optimizer = ["Adam", "AdamW", "SGD"]


epochs = 50
val_test_metric = "mse"
watch_losses = {"mse": {}}
main_loss = "mse"

# ---CONSTANTS---
seed = 42
