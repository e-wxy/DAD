import torch
import torch.nn as nn
import torch.distributions as dist
from .bed import BED
from .extra_dist import LowerTruncatedNormal


class DeathProcess(BED):
    """Simulate Death Process Experiment"""

    def __init__(
        self,
        theta_loc=None,         # prior on theta
        theta_scale=None,       # prior on theta
        theta_dist="truncated_normal", # prior distribution type
        N: int = 50,            # total number of people
        design_scale=None,
        outcome_scale=50,
    ):
        super(DeathProcess, self).__init__(design_scale, outcome_scale)
        # prior of theta
        self.theta_loc = theta_loc if theta_loc is not None else torch.tensor(1.0)
        self.theta_scale = theta_scale if theta_scale is not None else torch.tensor(1.0)
        if theta_dist == "truncated_normal":
            self.theta_prior_dist = LowerTruncatedNormal(
                self.theta_loc, self.theta_scale, 0.0
            )
        elif theta_dist == "lognormal":
            self.theta_prior = dist.LogNormal(self.theta_loc, self.theta_scale)
        else:
            raise ValueError(f"Invalid option: theta_dist={theta_dist}.")
        
        self.N = N
        self.outcome_scale = N

        self.softplus = nn.Softplus()

    @torch.no_grad()
    def sample_theta(self, size):
        """ Sample latent variable from the prior """
        theta = self.theta_prior.sample(size).unsqueeze(-1)
        theta = theta.clamp(min=1e-10, max=1e10)
        return theta

    def to_design_space(self, xi):
        """ Constrain the designs to the design space """
        xi = self.softplus(xi)
        return xi


    def forward(self, xi, theta):
        """ Experiment's outcome

        Args:
            xi [B, 1]: design
            theta [B, 1]: true rate

        Returns:
            y [B, 1]
        """
        death_prob = 1 - (-xi * theta).exp()
        # constrain the death probability to be in [0, 1]
        death_prob = torch.clamp(death_prob, 0.0, 1.0)
        y = dist.Binomial(total_count=self.N, probs=death_prob).sample()
        return y
    
    def log_likelihood(self, y, xi, theta):
        death_prob = 1 - (-xi * theta).exp()
        # constrain the death probability to be in [0, 1]
        death_prob = torch.clamp(death_prob, 0.0, 1.0)
        m = dist.Binomial(total_count=self.N, probs=death_prob)
        log_prob = m.log_prob(y)

        return log_prob
    
    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"DeathProcess({', '.join('{}={}'.format(key, val) for key, val in info.items())})"
