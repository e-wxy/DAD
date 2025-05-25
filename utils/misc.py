import torch
import os
import numpy as np
import random


def set_seed(seed: int):
    """ Set random seed for reproducibility """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model, dir, name="mlp.pth"):
    """ Save model state dict """
    file_path = os.path.join(dir, 'model')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, name)
    torch.save(model.state_dict(), file_path)
    return file_path


def load_state_dict(model, dir, name="mlp.pth"):
    """ Load model state dict """
    file_path = os.path.join(dir, 'model', name)
    model.load_state_dict(torch.load(file_path, map_location=torch.get_default_device(), weights_only=True))
    return model


def save_checkpoint(cfg, model, optimizer, scheduler, epoch):
    """ Save checkpoint """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        # random generator
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }

    checkpoint_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    torch.save(ckpt, checkpoint_path)

    return checkpoint_path


def load_checkpoint(cfg, model, optimizer, scheduler):
    """ Load checkpoint """
    ckpt = torch.load(os.path.join(cfg.output_dir, cfg.checkpoint_name), map_location=torch.get_default_device(), weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    epoch = ckpt["epoch"]
    # Restore RNG states
    if not isinstance(ckpt['rng_state'], torch.ByteTensor):
        ckpt['rng_state'] = torch.ByteTensor(ckpt['rng_state'].cpu())
    torch.set_rng_state(ckpt['rng_state'])

    if torch.cuda.is_available() and ckpt['cuda_rng_state'] is not None:
        if not isinstance(ckpt['cuda_rng_state'], torch.ByteTensor):
            ckpt['cuda_rng_state'] = torch.ByteTensor(ckpt['cuda_rng_state'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])

    np.random.set_state(ckpt['numpy_rng_state'])
    random.setstate(ckpt['random_rng_state'])

    return epoch
