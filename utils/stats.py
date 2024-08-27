import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal, kl_divergence
import warnings

# from utils.params import set_params

# class CDFNormal(nn.Module):
#     def __init__(self, loc=0, scale=1):
#         super().__init__()
#         self.loc=loc
#         self.scale=scale

#     def forward(self, input):
#         dist = torch.distributions.Normal(loc=self.loc, scale=self.scale)
#         return dist.cdf(input)

# class Gaussian(nn.Module):
#     """
#     Multi-variate Gaussian distribution with learnable parameters. 
#     """

#     def __init__(self, mu=0, sigma=1):
#         """
#         Parameters
#         ----------
#         mu          :
#         sigma       : (torch.tensor), 2D tensor corresponding to covariance
#                       matrix, or a 1D tensor that will be placed on the diagonal 
#                       of covariance matrix.
#         """
#         assert len(mu.shape) == 1, "`mu` should be a 1D tensor."
#         assert mu.shape[0] == sigma.shape[0], (
#             "Shapes of `mu` and `sigma` are not compatible."
#         )

#         self.mu = nn.Parameter(mu)
#         if len(sigma.shape) == 1:
#             self.sigma = nn.Parameter(torch.diag(sigma))
#         elif len(sigma.shape) == 2:
#             assert sigma.shape[0] == sigma.shape[1], (
#                 "If `sigma` is a 2D tensor, it must be square."
#             )
#             self.sigma = nn.Parameter(sigma)
#         else:
#             raise Exception("`sigma` should be a 1D or 2D tensor.")
        
#         self.dist = self.update_dist()

#     def update_dist(self):
#         """
#         Replace distribution attribute, using current parameters, e.g. for
#         initialization or after optimization step.
#         """
#         self.dist = torch.distributions.MultivariateNormal(self.mu, self.sigma)
    
#     def rsample(self):
#         """
#         Sample using the reparameterization trick.
#         """
#         x = self.dist.rsample()
#         return x

class Gaussian(nn.Module):
    """
    Multi-variate Gaussian distribution with learnable parameters. 
    """

    def __init__(self, optimize='diag', **kwargs):
        """
        Parameters
        ----------
        optimize : (String), either make passed in arguments parameters first, then construct mu and Sigma, or construct mu and Sigma first, then make results parameters.
        mu_1 
        mu_2 
        ...
        mu_n
        sigma_1
        sigma_2
        ...
        sigma_n : (torch.tensor), 2D tensor corresponding to covariance matrix,
                  or a 1D tensor that will be placed on the diagonal of 
                  covariance matrix.
        """
        store single param as nn.Parmater, then make copies etc.
        if optimize == 'diag':
            # This needs to be stored as attribute in order to average partial
            # derivatives.
            self.separate_params = {
                key : nn.Parameter(value) for key, value in kwargs.items()
            }
            self.mu, self.sigma = construct_mu_sigma(self.separate_params)
        elif optimize == 'full':
            # First concatenate everything to build full mean vector and 
            # covariance matrix, then set as parameters.
            mu, sigma = construct_mu_sigma(kwargs)
            self.mu, self.sigma = nn.Parameter(mu), nn.Parameter(sigma)
        else:
            raise Exception("Invalid value for `optimize`.")
        
        self.dist = self.update_dist()

    def update_dist(self):
        """
        Replace distribution attribute, using current parameters, e.g. for
        initialization or after optimization step.
        """
        self.dist = torch.distributions.MultivariateNormal(self.mu, self.sigma)
    
    def rsample(self):
        """
        Sample using the reparameterization trick.
        """
        x = self.dist.rsample()
        return x

    

def construct_mu_sigma(mu_sigma, validate_args=True):
    """
       
    """
  
    if validate_args:
        assert (
            sum(1 for key in mu_sigma.keys() if 'mu' in key) 
            == sum(1 for key in mu_sigma.keys() if 'sigma' in key)
        ), (
            "Number of `mu` parameters and `sigma` parameters must match."
        )

    # Extract mu and sigma tensors, put sigma on diagonal of matrix if is 1D (if
    # already is 2D, leave in place).
    mu_tuple = tuple([tensor for key, tensor in mu_sigma.items() if 'mu' in key])
    sigma_list = [
        torch.diag(tensor) if len(tensor.shape) == 1 else tensor 
        for key, tensor in mu_sigma.items() if 'sigma' in key
    ]

    mu = torch.cat(mu_tuple)
    sigma = torch.block_diag(*sigma_list)

    return mu, sigma
 

def binomial_log_likelihood(total_count, correct, probs):
    """
    Compute joint log likelihood across multiple binomial outcomes (frame pairs).

    Parameters
    ----------
    total_count : Tensor of total trials for each frame pair.
    correct     : Tensor of number of correct for each frame pair.
    probs       : Tensor of probability of correct response for each frame pair.
    """
    binom = torch.distributions.binomial(
        total_count=total_count,
        probs=probs
    )
    log_likelihood = torch.sum(binom.log_probs(correct))
    return log_likelihood


def normal_cdf(x, mu=0, sigma2=1):
    """
    Evaluate Gaussian CDF with specified mean and variance at supplied value.
    """
    dist = torch.distributions.Normal(loc=mu, scale=sigma2)
    eval = dist.cdf(x)
    return eval


def compute_mvn_kl_div_from_param(mu1, sigma1, mu2, sigma2):
    """
    """
    dist1, dist2 = MultivariateNormal(mu1, sigma1), MultivariateNormal(mu2, sigma2)
    return kl_divergence(dist1, dist2)


# def compute_mvn_kl_div_from_dist(dist1, dist2):
#     """
#     """
#     kl_div = kl_divergence(dist1, dist2)
#     return kl_div


def combine_scalar_cov(cov):
    """
    Function to compute variance of linear combination of possibly correlated 
    random variables.
    """
    assert len(cov.shape) == 2, "cov should be a 2D tensor."
    n = cov.shape[0]
    out = torch.sum(cov) / (n**2)
    return out


def combine_vector_cov(cov, block_ind):
    """
    Function to compute co-variance matrix of linear combination of
    multi-variate random variables.
    https://stats.stackexchange.com/questions/216163/covariance-matrix-for-a-linear-combination-of-correlated-gaussian-random-variabl
    """
    assert len(cov.shape) == 2, "cov should be a 2D tensor."
    n_vars = len(block_ind)

    if n_vars == 1:
        warnings.warn("Only one r.v. passed in, returning input as output.")
        return cov
    
    n_dims = block_ind[1] - block_ind[0]
    cross_cov_mats = torch.empty(n_vars, n_dims, n_dims)

    for 





