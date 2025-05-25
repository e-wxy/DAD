import torch
import torch.nn as nn
import torch.distributions as dist

from .bed import BED


class HiddenLocation(BED):
    """Simulate Location Finding Experiment"""

    def __init__(
        self,
        dim: int = 1,               # dimension of location
        K: int = 2,                 # number of souce points
        theta_p1=None,              # prior on theta
        theta_p2=None,              # prior on theta
        theta_dist="normal",        # prior distribution type
        design_scale=None,             # scale of the design space
        outcome_scale=None,           # scale of the experiment outcomes
        noise_scale=0.5,
        base_signal: float = 0.1,   # param of signal
        max_signal: float = 1e-4,   # param of signal
    ) -> None:
        super(HiddenLocation, self).__init__(design_scale, outcome_scale)
        # prior of theta
        self.theta_dist = theta_dist
        if theta_dist == "normal":
            # loc
            self.theta_p1 = theta_p1 if theta_p1 is not None else torch.zeros((K, dim))
            # covmat
            self.theta_p2 = theta_p2 if theta_p2 is not None else torch.eye(dim)
            if dim == 1:
                self.theta_prior = dist.Normal(self.theta_p1, self.theta_p2)
            else:
                self.theta_prior = dist.MultivariateNormal(
                    self.theta_p1, self.theta_p2
                )
        elif theta_dist == "uniform":
            # low
            self.theta_p1 = theta_p1 if theta_p1 is not None else torch.zeros((K, dim))          # low
            # high
            self.theta_p2 = theta_p2 if theta_p2 is not None else torch.ones((K, dim))  # scale: high - low
            self.theta_prior = dist.Uniform(
                self.theta_p1, self.theta_p2
            )
        else:
            raise ValueError(f"Prior distribution type {theta_dist} is not supported!")

        # scale of design space
        self.design_scale = (
            design_scale if design_scale is not None else torch.max(self.theta_p2)
        )
        # signal params
        noise_scale = noise_scale * torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("noise_scale", noise_scale)
        self.base_signal = base_signal
        self.max_signal = max_signal
        self.dim = dim
        self.K = K

        self.conditional = dist.Normal(0, self.noise_scale)


    @torch.no_grad()
    def sample_theta(self, size):
        """ Sample latent variable from the prior """
        return self.theta_prior.sample(size)

    def total_density(self, xi, theta):
        """Total density

        Shape:
            xi: [:, D] - [B, D] or [1/L, B, T, D]
            theta: [:, K, D] - [B, K, D] or [L, B, T, K, D]

        Returns:
            density: [:, 1]
        """
        # two norm squared
        sq_two_norm = (
            (xi.unsqueeze(-2).expand(theta.shape) - theta).pow(2).sum(axis=-1)
        )  # [:, K]
        sq_two_norm_inverse = (self.max_signal + sq_two_norm).pow(-1)
        # sum over the K sources, add base signal and take log.
        density = torch.log(
            self.base_signal + sq_two_norm_inverse.sum(-1, keepdim=True)
        )  # [:, 1]

        return density

    def to_design_space(self, xi):
        return xi

    def forward(self, xi, theta):
        """ Experiment's outcome
            Using Differentiable Sampling for Reparameterization Trick

        Args:
            xi [B, D]: normalised design
            theta [B, K, D]: sources

        Returns:
            observations: [B, 1]
        """
        signal = self.total_density(xi, theta)  # [B, 1]
        # add noise
        noised_signal = dist.Normal(signal, self.noise_scale).rsample()
        return noised_signal

    def log_likelihood(self, y, xi, theta):
        """Log likelihood from gaussian noise

        Args:
            y [:, 1]
            xi [:, D]: real designs
            theta [:, K, D]

        Returns:
            log_prob: log likelihoods, [:, 1]
        """
        # uncorrupted signal
        signal = self.total_density(xi, theta)
        # calculate the log likelihood
        log_prob = dist.Normal(signal, self.noise_scale).log_prob(y)
        return log_prob
    
    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"HiddenLocation({', '.join('{}={}'.format(key, val) for key, val in info.items())})"
