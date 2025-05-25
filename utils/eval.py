import torch
import numpy as np
from attrdictionary import AttrDict
from loss.eig import PCELoss, EIGStepLoss


@torch.no_grad()
def compute_EIG_from_history(experiment, theta_0, x, y, L=int(1e6), batch_size=40, stepwise=False):
    """ Evaluate the lower and upper bounds of sEIG from a minibatch of the history

    Args:
        theta_0 (torch.Tensor) [B, (K, )D]: initial theta
        x (torch.Tensor) [B, T, D_x]: history of designs
        y (torch.Tensor) [B, T, D_y]: history of outcomes
        T (int): number of proposed designs in a trajectory
        L (int): number of contrastive samples
        batch_size (int): mini batch size of outer samples
    """
    T = x.shape[1]

    criterion = EIGStepLoss(L, batch_size, experiment.log_likelihood, reduction='none')

    pce_losses = []
    nmc_losses = []

    thetas = experiment.sample_theta((L, batch_size))
    thetas = torch.concat([theta_0.unsqueeze(0), thetas], dim=0)          # [L+1, B, (K, )D]

    if stepwise:
        for t in range(T):
            pce_loss, nmc_loss = criterion(y[:, t], x[:, t], thetas)  # [B]
            pce_losses.append(pce_loss)
            nmc_losses.append(nmc_loss)
                
        pce_losses = torch.stack(pce_losses, dim=-1)  # [B, T]
        nmc_losses = torch.stack(nmc_losses, dim=-1)  # [B, T]
    else:
        for t in range(1, T + 1):
            pce_losses, nmc_losses = criterion(y[:, :t], x[:, :t], thetas)  # [B]

    # Calculate bounds
    pce_losses = torch.log(torch.tensor(L + 1)) - pce_losses  # [B(, T)]
    nmc_losses = torch.log(torch.tensor(L)) - nmc_losses      # [B(, T)]  

    return pce_losses, nmc_losses


@torch.no_grad()
def eval_bounds(cfg, model, experiment, T: int = 30, L: int = int(1e6), M: int = 2000, batch_size: int = 40, stepwise: bool = True):
    """ Evaluate the lower and upper bounds of sEIG

    Args:
        cfg: configuration
        model (nn.Module): design policy network
        experiment (BED): experiment data simulator
        T (int): number of steps in a trajectory
        L (int): number of contrastive samples
        M (int): number of parallel experiments for outer expectation
        batch_size (int): minibatch size for running on machines with low memories
        stepwise (bool): whether retain the sEIG bounds for each step

    Returns:
        AttrDict: 
            pce_mean (torch.Tensor) [T] or [1]: mean of sPCE bound
            pce_se (torch.Tensor) [T] or [1]: s.e. of sPCE bound
            nmc_mean (torch.Tensor) [T] or [1]: mean of sNMC bound
            nmc_se (torch.Tensor) [T] or [1]: s.e. of sNMC bound
    """
    
    model.eval()

    max_step = (M + batch_size - 1) // batch_size

    pce_list = []
    nmc_list = []

    for _ in range(max_step):
        # Run traces
        theta, xi_designs, y_outcomes = model.run_trace(experiment, T, batch_size)

        # Calcuate losses
        pce_losses, nmc_losses = compute_EIG_from_history(
            experiment, theta, xi_designs, y_outcomes, L, batch_size, stepwise
        )

        pce_list.append(pce_losses)
        nmc_list.append(nmc_losses)

    # Stack bounds
    pce = torch.cat(pce_list, dim=0)   # [M(, T)]
    nmc = torch.cat(nmc_list, dim=0)   # [M(, T)]

    # Calculate mean and s.e.
    M = pce.shape[0]
    pce_mean = torch.mean(pce, dim=0)    # [T]
    pce_se = torch.std(pce, dim=0) / np.sqrt(M)     # [T]
    nmc_mean = torch.mean(nmc, dim=0)    # [T]
    nmc_se = torch.std(nmc, dim=0) / np.sqrt(M)     # [T]

    pce_mean = pce_mean.cpu()
    pce_se = pce_se.cpu()
    nmc_mean = nmc_mean.cpu()
    nmc_se = nmc_se.cpu()

    bounds = AttrDict(pce_mean=pce_mean, pce_se=pce_se, nmc_mean=nmc_mean, nmc_se=nmc_se)

    return bounds


@torch.no_grad()
def eval_PCE(cfg, model, experiment, T=30, N=200, L=int(1e6), M=2000, batch_size=40):
    """ Evaluate the sPCE bound of sEIG """
    model.eval()

    pce_criterion = PCELoss(L, T, experiment.log_likelihood)
    max_step = (M + batch_size - 1) // batch_size

    pce_loss = 0

    for _ in range(max_step):
        # run traces
        theta, xi_designs, y_outcomes = model.run_trace(experiment, T, batch_size)

        # generate contrastive samples
        thetas = experiment.sample_theta((L, batch_size))
        thetas = torch.concat([theta.unsqueeze(0), thetas], dim=0)            # [L+1, B, K, D]

        # calcuate losses
        pce_loss += pce_criterion(y_outcomes, xi_designs, thetas) / max_step

    # Calculate bounds
    pce_loss = np.log(L + 1) - pce_loss.item()

    return pce_loss