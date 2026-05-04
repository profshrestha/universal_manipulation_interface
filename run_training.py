#!/usr/bin/env python
"""
UMI diffusion policy training pipeline.

Wraps train.py with sensible defaults for local and shared-folder use.

Usage:
  python run_training.py <zarr_path> [options]

Examples:
  # Basic — checkpoints saved to ~/sudhir_umi/checkpoints/<run_name>/
  python run_training.py ~/sudhir_umi/datasets/cube_hand_off/session/replay_buffer.zarr.zip

  # Custom run name and output dir
  python run_training.py replay_buffer.zarr.zip -n cube_handoff_v1 -o /path/to/checkpoints

  # Override batch size (default 64 for RTX 5090 / 32GB VRAM)
  python run_training.py replay_buffer.zarr.zip -bs 32

  # Single-arm dataset
  python run_training.py replay_buffer.zarr.zip -c train_diffusion_unet_timm_umi_workspace
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import subprocess
import datetime


@click.command()
@click.argument('zarr_path')
@click.option('-n', '--run_name', type=str, default=None,
    help='Name for this training run. Defaults to dataset filename + timestamp.')
@click.option('-o', '--output_dir', type=str, default=None,
    help='Root directory for checkpoints. Defaults to ~/sudhir_umi/checkpoints/')
@click.option('-c', '--config_name', type=str,
    default='train_diffusion_unet_umi_bimanual_workspace',
    help='Hydra config name. Default: train_diffusion_unet_umi_bimanual_workspace (bimanual). '
         'Use train_diffusion_unet_timm_umi_workspace for single-arm.')
@click.option('-bs', '--batch_size', type=int, default=32,
    help='Training batch size. Default 32 (bimanual: two ViT encoders ~2x memory). '
         'Reduce to 16 if you get OOM errors.')
@click.option('-vbs', '--val_batch_size', type=int, default=8,
    help='Validation batch size. Default 16.')
@click.option('-nw', '--num_workers', type=int, default=8,
    help='DataLoader worker count. Default 8.')
def main(zarr_path, run_name, output_dir, config_name, batch_size, val_batch_size, num_workers):
    zarr_path = pathlib.Path(os.path.expanduser(zarr_path)).absolute()
    if not zarr_path.is_file():
        print(f"Error: zarr file not found: {zarr_path}")
        sys.exit(1)

    # Build run name from dataset name + timestamp if not provided
    if run_name is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_stem = zarr_path.stem.replace('.zarr', '')
        run_name = f'{dataset_stem}_{ts}'

    # Default checkpoint root
    if output_dir is None:
        output_dir = pathlib.Path(os.path.expanduser('~/umi_checkpoints'))
    else:
        output_dir = pathlib.Path(os.path.expanduser(output_dir))

    checkpoint_dir = output_dir / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== UMI Training ===")
    print(f"  Dataset:     {zarr_path}")
    print(f"  Run name:    {run_name}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Config:      {config_name}")
    print(f"  batch_size:  {batch_size}  val_batch_size: {val_batch_size}  num_workers: {num_workers}")
    print()

    train_script = pathlib.Path(ROOT_DIR) / 'train.py'
    assert train_script.is_file(), f"train.py not found at {train_script}"

    cmd = [
        'python', str(train_script),
        f'--config-name={config_name}',
        f'task.dataset_path={zarr_path}',
        f'hydra.run.dir={checkpoint_dir}',
        f'dataloader.batch_size={batch_size}',
        f'val_dataloader.batch_size={val_batch_size}',
        f'dataloader.num_workers={num_workers}',
    ]

    env = os.environ.copy()
    env['WANDB_MODE'] = 'disabled'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nTraining failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nTraining complete. Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
