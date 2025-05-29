import argparse
import importlib
import os
import random
import torch
import numpy as np
import logging


def load_args():
    parser = argparse.ArgumentParser(
        description="Load config and print seq_path variable"
    )
    parser.add_argument("cfg_path", type=str, help="Path to the config file")
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Set seed to {seed} (torch, numpy, random)")


def import_config(config_path: str):
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, (
        f"Failed to load config from {config_path}"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_config(config_path: str = None):
    if config_path is None:
        args = load_args()
        config_path = args.cfg_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    config = import_config(config_path)
    return config


def setup_logger(output_dir: str):
    # 生成包含时间的日志文件名
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"{output_dir}/log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  # 写入文件
            logging.StreamHandler(),  # 保持控制台输出
        ],
    )
    logger = logging.getLogger(__name__)
    return logger
