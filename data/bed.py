import torch
import torch.nn as nn
import torch.distributions as dist


### generate the experiment outcomes
class BED(nn.Module):
    """Base class for simulating BED Experiment"""

    def __init__(
        self,
        design_scale=1,
        outcome_scale=1,
    ) -> None:
        super(BED, self).__init__()
        self.design_scale = design_scale
        self.outcome_scale = outcome_scale
        self.theta_prior = dist.Normal(0, 1)

    @torch.no_grad()
    def sample_theta(self, size):
        return self.theta_prior.sample(size)

    def to_design_space(self, xi):
        """ Constrain the designs to the design space """
        return xi
    
    def normalise_outcomes(self, y):
        if self.outcome_scale is None:
            return y
        else:
            return y / self.outcome_scale
    
    def normalise_design(self, x):
        if self.design_scale is None:
            return x
        else:   
            return x / self.design_scale

    @torch.no_grad()
    def forward(self, xi, theta):
        """Experiment's outcome

        Args:
            xi [B, D]: design
            theta [B, K, D]: latent varaibles to learn from the experiments

        Returns:
            observations: [B, 1]
        """
        raise NotImplementedError

    def log_likelihood(self, y, xi, theta):
        """Log likelihood from gaussian noise

        Args:
            y [B, 1]
            xi [B, D]: real designs
            theta [B, K, D]

        Returns:
            log_prob: log likelihoods, [B, 1]
        """
        raise NotImplementedError
