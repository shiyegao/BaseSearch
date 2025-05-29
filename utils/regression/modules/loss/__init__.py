from .mse_loss import MSE_Loss


loss_functions = {"mse": MSE_Loss}


def get_loss_function(name):
    if name not in loss_functions:
        raise ValueError(
            f"Loss function '{name}' is not defined. Available: {list(loss_functions.keys())}"
        )
    return loss_functions[name]
