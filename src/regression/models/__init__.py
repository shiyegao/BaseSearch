from .ConvNet import ConvNet1D

models = {
    "ConvNet": ConvNet1D,
}


def get_model(name):
    model_name = list(name.keys())[0]
    if model_name not in models:
        raise ValueError(
            f"Model '{name}' is not defined. Available: {list(models.keys())}"
        )
    return models[model_name](**name[model_name])
