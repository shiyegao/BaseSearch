import os
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys

sys.path.append(".")
from utils.regression.dataset.dataset import create_dataloader
from utils.tool import load_config


def evaluate_rf(model, dataloader):
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for inputs, labels in dataloader:
        inputs = inputs.numpy()
        labels = labels.numpy()

        predictions = model.predict(inputs)
        loss = np.mean((labels - predictions) ** 2)
        total_loss += loss

        all_labels.extend(labels)
        all_predictions.extend(predictions)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(config, n_estimators, max_depth):
    torch.manual_seed(config.seed)
    pkl_file_path = "data/pkl/dataset_Maize.pkl"
    train_loader, val_loader, test_loader = create_dataloader(pkl_file_path, config)

    X_train = []
    y_train = []
    for inputs, labels in train_loader:
        X_train.append(inputs.numpy())
        y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=config.seed,
        n_jobs=-1,
    )

    print(
        f"Training Random Forest Regressor with n_estimators={n_estimators}, max_depth={max_depth}..."
    )
    rf_model.fit(X_train, y_train)

    train_loss = evaluate_rf(rf_model, train_loader)
    val_loss = evaluate_rf(rf_model, val_loader)
    test_loss = evaluate_rf(rf_model, test_loader)

    print(
        f"Results - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}, Test MSE: {test_loss:.4f}"
    )

    return train_loss, val_loss, test_loss


if __name__ == "__main__":
    config = load_config("conf/regression/rf_config.py")
    n_estimatorss = config.n_estimators
    max_depths = config.max_depth

    results = []

    for n_estimators in n_estimatorss:
        for max_depth in max_depths:
            train_loss, val_loss, test_loss = main(config, n_estimators, max_depth)

            results.append(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "train_mse": train_loss,
                    "val_mse": val_loss,
                    "test_mse": test_loss,
                }
            )

    results_df = pd.DataFrame(results)
    os.makedirs("output/csv", exist_ok=True)
    results_path = f"output/csv/hyperparameter_results_{config.name}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"All hyperparameter results saved to {results_path}")
