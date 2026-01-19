""" Deep Adaptive Design for Location Finding """
import torch
import torch.nn as nn
import numpy as np
import os
import time
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from data import CES
from model.mlp import EncoderNetworkV2, EmitterNetwork, SetEquivariantDesignNetwork
from loss import PCELoss
from utils import create_logger, set_seed, save_checkpoint, save_state_dict
from utils.eval import eval_bounds, eval_PCE




def train(cfg, logger, model, experiment, batch_size, L, M, T, max_epoch, verbose=10):
    """ DAD Train

    Args:
        model: design policy network
        experiment: agent for simulating the experiment results
        batch_size (int): minibatch size of training
        L (int): number of contrastive samples for inner expectation
        M (int): number of parallel experiments for outer expectation, actual size is batch_size * max_step
        T (int): number of steps in each sequential experiment
        max_epoch (int): max epoch of optimisation
        verbose (int): periods to show the training process
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.8, 0.998), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    criterion = PCELoss(L, T, experiment.log_likelihood)

    losses = []

    max_step = (M + batch_size - 1) // batch_size

    total_training_time = []


    for epoch in range(max_epoch):
        start_time = time.time()
        model.train()
        loss = 0
        optimizer.zero_grad()

        # minibatching for low memory
        for step in range(max_step):
            # sample true theta
            theta_0 = experiment.sample_theta((batch_size,))                        # [B, K, D]

            # history of an experiment (use lists to avoid in-place operations)
            xi_list = []
            y_list = []

            # T-steps experiment
            for t in range(T):
                if t == 0:
                    xi_history = torch.empty((batch_size, 0, cfg.data.dim_design))
                    y_history = torch.empty((batch_size, 0, 1))
                else:
                    xi_history = torch.stack(xi_list, dim=1)                        # [B, t, D_x]
                    y_history = torch.stack(y_list, dim=1)                          # [B, t, 1]

                xi = model.forward(xi_history, y_history)                           # [B, D]
                y = experiment(experiment.to_design_space(xi), theta_0)             # [B, 1]

                xi_list.append(xi)
                y_list.append(y)

            xi_designs = torch.stack(xi_list, dim=1)                                # [B, T, D_x]
            y_outcomes = torch.stack(y_list, dim=1)                                 # [B, T, 1]


            # sample contrastive samples
            thetas = experiment.sample_theta((L, batch_size))
            thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)            # [L+1, B, K, D]
            batch_loss = criterion(y_outcomes, experiment.to_design_space(xi_designs), thetas) / max_step
            batch_loss.backward()
            loss += batch_loss.detach()

        losses.append(loss.item())

        # gradient clipping
        if cfg.clip_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type="inf")
            
        optimizer.step()

        if epoch % 1000 == 0:
            scheduler.step()

        if cfg.wandb.use_wandb:
            wandb.log({"loss": loss}, step=epoch)

        end_time = time.time()
        training_time = end_time - start_time
        total_training_time.append(training_time)

        if epoch % verbose == 0:
            pce_loss = eval_PCE(cfg, model, experiment, T, L, M, batch_size=cfg.eval_size)
            logger.info(f"Epoch: {epoch}, loss: {losses[-1]:.4f}, sPCE bound: {pce_loss}")
            if cfg.wandb.use_wandb:
                wandb.log({"sPCE": pce_loss}, step=epoch)

        if cfg.checkpoint and epoch % cfg.checkpoint == 0:
            logger.info(f"Checkpoint has been saved at {save_checkpoint(cfg, model, optimizer, scheduler, epoch + 1)}")

    logger.info(f"Training time: {np.sum(total_training_time) / 3600:.3f} hours in total, {np.mean(total_training_time) / 3600:.3f} hours per epoch")



@hydra.main(version_base=None, config_path="./config", config_name="ces")
def main(cfg):
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name='ces')

    # Setting device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.set_default_device(cfg.device)

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    logger.info("Running with config:\n{}".format(cfg))


    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # Setting random seed
    if cfg.fix_seed:
        set_seed(cfg.seed)

    
    # Model
    if cfg.model.name == 'MLP':
        encoder = EncoderNetworkV2(cfg.data.dim_design, cfg.data.dim_outcome, embed_dim=cfg.model.embed_dim, hidden_dim=cfg.model.hidden_dim, encoding_dim=cfg.model.encoding_dim, hidden_depth=cfg.model.encoder_hidden_depth, activation=nn.ReLU(), normalization=cfg.model.encoder_normalization)
        emitter = EmitterNetwork(cfg.model.encoding_dim, cfg.data.dim_design, hidden_dim=cfg.model.hidden_dim, hidden_depth=cfg.model.emitter_hidden_depth, activation=nn.ReLU())
        model = SetEquivariantDesignNetwork(
            encoder, emitter, 
            cfg.data.dim_design, cfg.data.dim_outcome, 
            empty_value=torch.ones(cfg.model.encoding_dim) * 0.01
        )
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")
    logger.info(model)
    
    # log gradients
    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=10)

    # Data
    experiment = CES(dim_design=cfg.data.dim_design, dim_outcome=cfg.data.dim_outcome, design_scale=cfg.data.design_scale, noise_scale=cfg.data.noise_scale)
    logger.info(experiment)
    
    # Train
    torch.autograd.set_detect_anomaly(True)
    train(cfg, logger, model, experiment, cfg.batch_size, cfg.L, cfg.M, cfg.T, cfg.max_epoch, verbose=cfg.verbose)


    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")

    # Evaluation
    bounds = eval_bounds(cfg, model, experiment, cfg.eval_T, cfg.eval_L, cfg.eval_M, cfg.eval_batch_size)
    logger.info(bounds)
    logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_se'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_se'][cfg.T-1]:.3f}")
    if cfg.wandb.use_wandb:
        wandb.log({"sPCE": bounds["pce_mean"][cfg.T-1], "sNMC": bounds["nmc_mean"][cfg.T-1]})

    if not os.path.exists(os.path.join(cfg.output_dir, "eval")):
        os.makedirs(os.path.join(cfg.output_dir, "eval"))
    torch.save(bounds, os.path.join(cfg.output_dir, "eval", "loc_dad_bounds.tar"))



if __name__ == '__main__':
    main()