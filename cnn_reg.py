import os
import torch
import torch.optim as optim
from itertools import product
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append(".")

from src.regression.dataset.dataset import create_dataloader
from src.tool import load_config
from src.regression.models import get_model
from src.regression.modules.loss import get_loss_function


class Trainer:
    def __init__(self, cfg_path=None):
        self.cfg = load_config(cfg_path)
        self.dynamic_keys = self._detect_dynamic_keys()

        self.species = self.cfg.species
        self.data_path = f"data/pkl/dataset_{self.species}.pkl"
        self.model_dir = ""

        self.csv_path = f"output/csv/{self.species}-as"
        self.ckpt_path = f"output/pth/{self.species}-as"
        self.one_epoch_val_results_name = f"one_epoch_val_results_{self.species}-as.csv"
        self.output_csv_name = f"results_{self.species}-as.csv"
        self.detailed_output_csv_name = f"detailed_results_{self.species}-as.csv"

        torch.manual_seed(self.cfg.seed)

    def _detect_dynamic_keys(self):
        dynamic_keys = [
            key for key, value in self.cfg.__dict__.items() if isinstance(value, list)
        ]
        return dynamic_keys

    def _generate_combinations(self):
        dynamic_values = {key: getattr(self.cfg, key) for key in self.dynamic_keys}
        dynamic_keys = list(dynamic_values.keys())
        dynamic_value_lists = list(dynamic_values.values())

        combinations = list(product(*dynamic_value_lists))

        combination_dicts = []
        for combo in combinations:
            combo_dict = {dynamic_keys[i]: combo[i] for i in range(len(dynamic_keys))}
            combination_dicts.append(combo_dict)

        return combination_dicts

    def _update_config(self, combo_dict):
        for key, value in combo_dict.items():
            setattr(self.cfg, key, value)

    def _initialize_from_config(self):
        self.model_name = self.cfg.model_name
        self.backbone_name = getattr(self.cfg, "backbone_name", None)

        self.epochs = self.cfg.epochs
        self.batch_size = self.cfg.batch_size
        self.learning_rate = self.cfg.learning_rate
        self.weight_decay = self.cfg.weight_decay
        self.optimizer = self.cfg.optimizer

        self.watch_losses = self.cfg.watch_losses
        self.main_loss = self.cfg.main_loss
        self.val_test_metric = self.cfg.val_test_metric

        self.project_name = self.cfg.project_name
        self.run_name = self.cfg.name

    def _initialize_components(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.model_name).to(self.device)

        self.train_loader, self.val_loader, self.test_loader = create_dataloader(
            self.data_path, self.cfg
        )

        optimizer_class = getattr(optim, self.optimizer)
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.loss_fn_dict = {
            name: get_loss_function(name)(**self.watch_losses.get(name, {}))
            for name in self.watch_losses.keys()
        }

    def _forward_pass(self, inputs):
        return self.model(inputs)

    def train_one_epoch(self):
        self.model.train()
        total_losses = {name: 0.0 for name in self.loss_fn_dict}

        for inputs, labels in self.train_loader:
            labels = labels.to(self.device, dtype=torch.float32)
            inputs = inputs.to(self.device, dtype=torch.float32)
            # import pdb; pdb.set_trace()
            outputs = self._forward_pass(inputs)

            losses = {
                name: loss_fn(outputs, labels)
                for name, loss_fn in self.loss_fn_dict.items()
            }

            main_loss = losses[self.main_loss]
            self.optimizer.zero_grad()
            main_loss.backward()
            self.optimizer.step()

            for name, loss_value in losses.items():
                total_losses[name] += loss_value.item()

        avg_losses = {
            name: total / len(self.train_loader) for name, total in total_losses.items()
        }
        return avg_losses

    def evaluate(self, dataloader):
        self.model.eval()
        total_losses = {name: 0.0 for name in self.loss_fn_dict}
        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.to(self.device)
                inputs = inputs.to(self.device)
                outputs = self._forward_pass(inputs)

                losses = {
                    name: loss_fn(outputs, labels)
                    for name, loss_fn in self.loss_fn_dict.items()
                }
                for name, loss_value in losses.items():
                    total_losses[name] += loss_value.item()

        avg_losses = {
            name: total / len(dataloader) for name, total in total_losses.items()
        }
        return avg_losses

    @staticmethod
    def model_name_to_str(model_name_dict):
        return "_".join(
            f"{model}_"
            + "_".join(
                f"{k}={'x'.join(map(str, v)) if isinstance(v, list) else v}"
                for k, v in params.items()
            )
            for model, params in model_name_dict.items()
        )

    def save_model(self, combo, epoch):
        self.model_dir = os.path.join(
            self.ckpt_path,
            self.model_name_to_str(self.model_name),
            "_".join(f"{k}={v}" for k, v in combo.items() if k != "model_name"),
        )
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model parameters saved to {model_path}")

    def log_metrics(self, epoch, losses):
        loss_logs = {
            f"{phase}_{name}": value
            for phase, loss_dict in losses.items()
            for name, value in loss_dict.items()
        }
        loss_logs.update({"epoch": epoch})

    def save_one_epoch_val_results(
        self, combo, val_losses_by_epoch, train_losses_by_epoch
    ):
        one_epoch_val_results_path = os.path.join(
            self.csv_path,
            self.model_name_to_str(self.model_name),
            "_".join(f"{k}={v}" for k, v in combo.items() if k != "model_name"),
        )
        os.makedirs(one_epoch_val_results_path, exist_ok=True)
        df = pd.DataFrame(
            {
                "Epoch": list(val_losses_by_epoch.keys()),
                "Validation Loss": list(val_losses_by_epoch.values()),
                "Training Loss": list(train_losses_by_epoch.values()),
            }
        )
        # 按行保存到文件
        one_epoch_val_results_file = os.path.join(
            one_epoch_val_results_path, self.one_epoch_val_results_name
        )
        df.to_csv(one_epoch_val_results_file, index=False)

    def train_and_evaluate(self):
        combinations = self._generate_combinations()
        results = []  # save the best val loss and test loss
        detailed_results = []  # save all the results

        print(f"Total combinations: {len(combinations)}")

        for idx, combo in enumerate(combinations, 1):
            # ----------------------------------------------------------------
            #                            select one combination
            # ----------------------------------------------------------------
            print(f"\nTraining combination {idx}/{len(combinations)}: {combo}")

            # update the config and initialize the components
            self._update_config(combo)
            self._initialize_from_config()
            self._initialize_components()

            # ----------------------------------------------------------------
            #                                train
            # ----------------------------------------------------------------
            val_losses_by_epoch = {}
            train_losses_by_epoch = {}
            best_val_loss = float("inf")
            best_epoch = None

            # start training
            for epoch in tqdm(
                range(1, self.cfg.epochs + 1), desc=f"Training combo {idx}"
            ):
                train_losses = self.train_one_epoch()
                val_losses = self.evaluate(self.val_loader)

                train_main_loss = train_losses[self.main_loss]
                train_losses_by_epoch[epoch] = train_main_loss

                val_main_loss = val_losses[self.val_test_metric]
                val_losses_by_epoch[epoch] = val_main_loss

                if val_main_loss < best_val_loss:
                    best_val_loss = val_main_loss
                    best_epoch = epoch

                self.log_metrics(epoch, {"train": train_losses, "val": val_losses})
                self.save_model(combo, epoch)

            print(
                f"Best epoch for combination {idx}: {best_epoch} with val loss {best_val_loss:.4f}"
            )
            # save the val loss of each epoch
            self.save_one_epoch_val_results(
                combo, val_losses_by_epoch, train_losses_by_epoch
            )

            # ----------------------------------------------------------------
            #           test the best model (best_epoch, min val_loss)
            # ----------------------------------------------------------------

            # load the best model parameters
            best_model_path = os.path.join(self.model_dir, f"epoch_{best_epoch}.pth")
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded model parameters from best epoch: {best_epoch}")

            # test the best model
            print("Starting testing...")
            test_losses = self.evaluate(self.test_loader)

            # print the val and test results
            print(f"Test Results for Best Epoch ({best_epoch}):")
            for loss_name, loss_value in test_losses.items():
                print(f"  {loss_name}: {loss_value:.4f}")

            # record the results to the dictionary
            result_row = {**combo}
            result_row["best_epoch"] = best_epoch
            result_row["best_val_loss"] = best_val_loss
            for loss_name, loss_value in test_losses.items():
                result_row[f"test_{loss_name}"] = loss_value
            for loss_name, loss_value in val_losses.items():
                result_row[f"val_{loss_name}"] = loss_value
            result_row["model_dir"] = self.model_dir

            results.append(result_row)

        # ----------------------------------------------------------------
        #          save the preliminary results to the csv file
        # ----------------------------------------------------------------
        output_csv_path = os.path.join(
            self.csv_path,
            self.output_csv_name,
        )
        detailed_output_csv_path = os.path.join(
            self.csv_path,
            self.detailed_output_csv_name,
        )

        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"\nInitial results saved to {output_csv_path}")

        # sort the results by the average of the val and test losses
        df = df.sort_values(by="best_val_loss").reset_index(drop=True)

        # find the optimal combination and epoch
        optimal_combo = df.iloc[0]
        optimal_params = optimal_combo.to_dict()
        optimal_epoch = self.cfg.epochs

        print(f"\nOptimal parameters: {optimal_params}")
        print(f"Optimal epoch: {optimal_epoch}")

        # ----------------------------------------------------------------
        #   test and val data are unified to the optimal parameters' epoch
        # ----------------------------------------------------------------
        for idx, row in df.iterrows():
            combo = row.to_dict()
            epoch = optimal_epoch

            self.model_dir = row["model_dir"]

            self.model_name = row["model_name"]
            self.model = get_model(self.model_name).to(self.device)
            # import pdb; pdb.set_trace()
            self.model.load_state_dict(
                torch.load(os.path.join(self.model_dir, f"epoch_{epoch}.pth"))
            )

            print(f"\nEvaluating combination {idx + 1}/{len(df)} at epoch {epoch}...")
            test_losses = self.evaluate(self.test_loader)

            val_losses = self.evaluate(self.val_loader)

            columns_to_remove = ["best_epoch", "best_val_loss", "model_dir"]

            filtered_combo = {
                key: value
                for key, value in combo.items()
                if key not in columns_to_remove
            }
            detailed_row = {**filtered_combo}
            for loss_name, loss_value in test_losses.items():
                detailed_row[f"test_{loss_name}"] = loss_value
            for loss_name, loss_value in val_losses.items():
                detailed_row[f"val_{loss_name}"] = loss_value
            detailed_row["epoch"] = epoch
            detailed_results.append(detailed_row)

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(detailed_output_csv_path, index=False)
        print(f"\nDetailed results saved to {detailed_output_csv_path}")

        return df, detailed_df


if __name__ == "__main__":
    cfg_path = "conf/regression/cnn_config.py"

    trainer = Trainer(cfg_path)
    trainer.train_and_evaluate()
