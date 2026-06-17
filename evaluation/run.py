#!/usr/bin/env python3

import argparse
import os
import random

import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import evaluate_agent, nonlearning_inference


def _patch_transformers_normalize() -> None:
    import transformers.image_processing_utils as image_processing_utils
    import transformers.image_transforms as image_transforms
    from transformers.image_transforms import (
        get_channel_dimension_axis,
        infer_channel_dimension_format,
        to_channel_dimension_format,
    )

    def safe_normalize(
        image: np.ndarray,
        mean,
        std,
        data_format=None,
        input_data_format=None,
        **kwargs,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
        num_channels = image.shape[channel_axis]

        image = np.array(image, dtype=np.float32, copy=True, order="C")
        mean = np.asarray([mean] * num_channels if np.isscalar(mean) else mean, dtype=np.float32)
        std = np.asarray([std] * num_channels if np.isscalar(std) else std, dtype=np.float32)

        if mean.size != num_channels:
            raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {mean.size}")
        if std.size != num_channels:
            raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {std.size}")

        broadcast_shape = [1] * image.ndim
        broadcast_shape[channel_axis] = num_channels
        image = (image - mean.reshape(broadcast_shape)) / std.reshape(broadcast_shape)

        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_data_format)
        return image

    image_transforms.normalize = safe_normalize
    image_processing_utils.normalize = safe_normalize


_patch_transformers_normalize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, num_chunks: int, chunk_idx: int, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
        nonlearning_inference(config)
        return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"

    trainer = trainer_init(config, num_chunks, chunk_idx)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()
