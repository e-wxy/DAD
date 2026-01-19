""" Custom distributions for PyTorch """
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints, Normal, TransformedDistribution, SigmoidTransform
from torch.distributions.utils import broadcast_all

def is_bad(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


class LowerTruncatedNormal(Distribution):
    r"""
    A Normal distribution truncated from below.

    Example::

        >>> m = LowerTruncatedNormal(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([-1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
        truncation (float or Tensor): point to truncate the Normal
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'truncation': constraints.real}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, truncation, validate_args=None):
        self.loc, self.scale, self.truncation = broadcast_all(loc, scale, truncation)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(LowerTruncatedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LowerTruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.truncation = self.truncation.expand(batch_shape)
        super(LowerTruncatedNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        raw_samples = self.icdf(u)
        samples = torch.nn.functional.relu(raw_samples - self.truncation) + self.truncation
        return samples

    def _normal_log_prob(self, value):
        var = self.scale ** 2
        log_scale = math.log(self.scale.item()) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._normal_log_prob(value) - self._normal_cdf(2 * self.loc - self.truncation).log()

    def _normal_cdf(self, value):
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def _normal_icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (self._normal_cdf(value) - self._normal_cdf(self.truncation)).clamp(min=0.)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        cdf_truncation = self._normal_cdf(self.truncation)
        rescaled_value = cdf_truncation + (1. - cdf_truncation) * value
        return self._normal_icdf(rescaled_value)


class CensoredSigmoidNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "lower_lim": constraints.real,
        "upper_lim": constraints.real,
    }

    has_rsample = True  # Enables reparameterized sampling

    def __init__(self, loc, scale, lower_lim, upper_lim, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        self.upper_lim, self.lower_lim = broadcast_all(upper_lim, lower_lim)

        self.normal = Normal(self.loc, self.scale)
        self.transform = SigmoidTransform()
        self.base_dist = TransformedDistribution(self.normal, [self.transform])

        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def z(self, value):
        return (self.transform.inv(value) - self.loc) / self.scale

    @property
    def support(self):
        return constraints.interval(self.lower_lim.min(), self.upper_lim.max())

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        x = self.rsample(sample_shape)
        return x

    def rsample(self, sample_shape=torch.Size()):
        x = self.base_dist.rsample(sample_shape)
        return torch.minimum(torch.maximum(x, self.lower_lim), self.upper_lim)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Broadcast all inputs to same shape
        value, upper_lim, lower_lim = torch.broadcast_tensors(value, self.upper_lim, self.lower_lim)

        log_prob = self.base_dist.log_prob(value)

        # Compute log_cdf values at limits
        upper_cdf = 1. - self.base_dist.cdf(upper_lim)
        lower_cdf = self.base_dist.cdf(lower_lim)

        crit = 2 * torch.finfo(value.dtype).tiny

        mask_upper = upper_cdf < crit
        mask_lower = lower_cdf < crit

        z_upper = self.z(upper_lim)
        z_lower = self.z(lower_lim)

        asymptotic_upper = self.base_dist.log_prob(upper_lim) - (crit + z_upper.abs()).log()
        asymptotic_lower = self.base_dist.log_prob(lower_lim) - (crit + z_lower.abs()).log()

        if is_bad(asymptotic_upper[mask_upper]):
            raise ArithmeticError("NaN in asymptotic upper")
        if is_bad(asymptotic_lower[mask_lower]):
            raise ArithmeticError("NaN in asymptotic lower")

        upper_cdf = torch.where(mask_upper, torch.ones_like(upper_cdf), upper_cdf)
        lower_cdf = torch.where(mask_lower, torch.ones_like(lower_cdf), lower_cdf)

        upper_logcdf = upper_cdf.log()
        lower_logcdf = lower_cdf.log()

        upper_logcdf = torch.where(mask_upper, asymptotic_upper, upper_logcdf)
        lower_logcdf = torch.where(mask_lower, asymptotic_lower, lower_logcdf)

        # Fill in special values based on value mask
        log_prob = torch.where(value == upper_lim, upper_logcdf, log_prob)
        log_prob = torch.where(value == lower_lim, lower_logcdf, log_prob)
        log_prob = torch.where(value > upper_lim, float('-inf'), log_prob)
        log_prob = torch.where(value < lower_lim, float('-inf'), log_prob)

        if is_bad(log_prob):
            raise ArithmeticError("NaN in log_prob")

        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        cdf_val = self.base_dist.cdf(value)
        cdf_val = torch.where(value >= self.upper_lim, torch.ones_like(cdf_val), cdf_val)
        cdf_val = torch.where(value < self.lower_lim, torch.zeros_like(cdf_val), cdf_val)
        return cdf_val

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CensoredSigmoidNormal, _instance)

        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.upper_lim = self.upper_lim.expand(batch_shape)
        new.lower_lim = self.lower_lim.expand(batch_shape)

        new.normal = Normal(new.loc, new.scale)
        new.transform = self.transform
        new.base_dist = TransformedDistribution(new.normal, [new.transform])

        super(CensoredSigmoidNormal, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new