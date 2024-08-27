from abc import ABC, abstractmethod
import copy
import itertools
import math
import torch
import torch.nn as nn

# from ..utils.general import get_upper_tri_lin_ind
from ..utils.params import set_params
from ..utils.stats import binomial_log_likelihood, normal_cdf, kl_divergence


class FreeEnergy(ABC, nn.Module):
    """
    """

    default_params = {
        'prior_init' : {
            'mu' : {
                'd' : torch.tensor([0.2], dtype=torch.float32),
                'c' : torch.tensor([math.pi/2], dtype=torch.float32),
                'a' : torch.tensor([0], dtype=torch.float32),
                'l' : torch.tensor([0], dtype=torch.float32)
            },
            'sigma' : {
                'd' : torch.tensor([1], dtype=torch.float32),
                'c' : torch.tensor([1], dtype=torch.float32),
                'a' : torch.eye(10),
                'l' : torch.tensor([1], dtype=torch.float32)
            }
        },
        'var_init' : 'random',
        'rectifier' : 'softplus',
        'lambda_max' : 0.06,
        'precision' : torch.float32
    }

    def __init__(self, data, params, kwargs_lock=False, **kwargs):
        """
        """
        self.params = set_params(
            self.default_params, params, kwargs_lock=kwargs_lock, **kwargs
        )

        self.data = data
        
        # Number of dimensions for each latent.
        self.n_nodes = self.data.n_frames
        self.latents_dims = {
            'd' : self.n_nodes - 1,                        # Distance
            'c' : self.n_nodes - 2,                        # Curvature
            'a' : (self.n_nodes - 2) * (self.n_nodes - 1), # Acceleration
            'l' : 1                                        # Lapse rate
        }

        self.q = self.init_prior()
        self.prior = self.init_variational_post()
    
    def transforms(self, x, var):
        """
        """
        if var == 'd':
            y = self.rectifier(x, self.params.rectifier)
        elif var == 'c':
            # Return same tensor.
            y = x
        elif var == 'a':
            # For `a`, orthogonalization depends on previous displacement vector
            # and will hence be computed during trajectory construction. Return
            # the same tensor.
            y = x
        elif var == 'l':
            y = self.params.lamda_max * normal_cdf(x, mu=0, sigma2=1)
        else: 
            raise Exception("Unrecognized value for `var`.")
        
        return y

    def construct_trajectory(self, d, c, a):
        """

        Parameters
        ----------
        d : n_samples x (n_nodes - 1) tensor of transformed variational samples.
        c : n_samples x (n_nodes - 2) tensor of transformed variational samples.
        a : n_samples x (n_nodes - 2) x (n_nodes - 1) tensor of transformed
            variational samples.
        
        Returns
        -------
        x : n_samples x n_nodes x (n_nodes - 1) array corresponding to n_samples
            different N timepoint trajectories through N - 1 dimensional space,
            for N = n_nodes.
        """
        assert d.shape[0] == c.shape[0] == a.shape[0], (
            "All inputs should have the same first dimension size."
        )
        # -------------------------------------------------------
        def next_distance(x_prev, d, v_h): return x_prev + d * v_h
        # -------------------------------------------------------

        n_samples = d.shape[0]
        eff_dim = self.n_nodes - 1

        # Second dimension of v_h is of size n_nodes = (n_nodes - 1) + 1, which
        # corresponds to the n_nodes - 1 displacements, plus an undefined
        # displacement to the first location (for indexing purposes).
        v_h = torch.zeros(n_samples, self.params.n_nodes, eff_dim)
        v_h[:, 0, :] = torch.nan # Displacement to first point is undefined
        v_h[:, 1, 0] = 1 # First normalized disaplcement is first standard basis vector

        # Add dummy vector for first curvature/acceleration for indexing
        # purposes.
        a = torch.cat((torch.zeros(n_samples, 1, self.n_nodes - 1), a), dim=1)
        d = torch.cat((torch.zeros(n_samples, 1), d), dim=1)

        # Initialize x_0 at origin, and compute x_1 using first normalized
        # displacement vector.
        x = torch.zeros(v_h.shape)
        x[:, 1, :] = next_distance(x[:, 0, :], d[:, 1], v_h[:, 1, :])

        # Compute locations for remaining nodes.
        for i_node in range(2, self.params.n_nodes):
            # Orthogonalize.
            Q, _ = torch.linalg.qr(torch.cat(v_h[:, i_node-1, :], a[:, i_node-1, :], 1))
            a = Q[:, 1]

            # Compute current displacement vector.
            v_h[i_node, :] = (
                torch.cos(c[i_node-1]) * v_h[i_node - 1] + torch.sin(c[i_node-1]) * a
            )

            # Compute current location.
            x[:, i_node, :] = next_distance(
                x[:, i_node - 1, :], 
                d[:, i_node], 
                v_h[:, i_node, :]
            )
            # x[:, i_node, :] = x[:, i_node - 1, :] + d[:, i_node] * v_h[:, i_node, :]

        return x
    
    def compute_expected_log_likelihood(self):
        """
        """
        # -------------------------------------------------------
        def compute_paxb(pdists):
            """Compute the probability of """
            p_axb = (
                (normal_cdf(pdists / torch.sqrt(2), mu=0, sigma2=1) 
                * normal_cdf(pdists / 2, mu=0, sigma2=1))
                + 
                (normal_cdf(-pdists / torch.sqrt(2), mu=0, sigma2=1) 
                * normal_cdf(-pdists / 2, mu=0, sigma2=1))
                )
            return p_axb      
        
        def compute_p(p_axb, lapse):
            """Compute the probability of performing the task correctly."""
            p = (1 - 2 * lapse) * p_axb + lapse
            return p
        # -------------------------------------------------------

        locals = self.sample_local_vars()
        x = self.construct_trajectory(locals['d'], locals['c'], locals['a'])

        pdists = torch.cdist(x.transpose(), x.transpose())
        p_axb = compute_paxb(pdists)
        probs = compute_p(p_axb, locals['l'])

        log_likelihood = binomial_log_likelihood(
            self.data.all, 
            self.data.correct,
            probs
        )

        return log_likelihood

    def compute_loss(self):
        log_likelihood = self.compute_expected_log_likelihood()
        kl_div = self.compute_kl_div()
        self.loss = log_likelihood + kl_div 

    def backward(self):
        self.loss.backward()

    def update(self):
        """
        The Guassian class has as an attribute an object of the
        torch.distributions.MultivariateNormal class. After an optimization
        step, this method instantiates a new object with the updated parameters.
        """
        self.prior = self.update_prior()
        self.q = self.update_var_post()

    @abstractmethod
    def init_prior(self):
        pass

    @abstractmethod
    def init_variational_post(self):
        """
        """
        pass

    @abstractmethod
    def update_prior(self):
        pass

    @abstractmethod
    def update_var_post(self):
        pass

    @abstractmethod
    def sample_local_vars(self):
        """
        """
        pass

    @abstractmethod
    def compute_kl_div(self):
        """
        """
        pass

    @staticmethod
    def rectifier(x, method):
        """
        """
        if method == 'softplus':
            f = nn.Softplus()
            out = f(x)
        else:
            raise Exception("Unrecognized value for `method`.")
        return out
    
    @staticmethod
    def reshape_a(a, n_nodes):
        """
        After sampling, a will be of shape n_samples x (n_nodes - 1)(n_nodes -
        2), this operation reshapes a to be of shape n_samples x (n_nodes - 1) x
        (n_nodes - 2), i.e. n_samples matrices each of shape (effective dim x
        number of acceleration vectors).

        Parameters
        ----------
        a       : Tensor of variational samples corresponding to acceleration.
        n_nodes : Number of frames.
        """
        n_samples = a.shape[0]
        return torch.reshape(a, (n_samples, n_nodes - 1, n_nodes - 2))


class FreeEnergyA(FreeEnergy):
    """
    KL divergence on full variational posterior.
    """

    def __init__(self, data, params, kwargs_lock=False, **kwargs):
        """
        """
        super().__init__(data, params, kwargs_lock=kwargs_lock, **kwargs)

        # Prepare indices for local variables.
        cum_sum = list(itertools.accumulate(self.latents_dims.values()))
        cum_sum.insert(0, 0)
        self.local_vars_ind = {
            var : range(cum_sum[i_var], cum_sum[i_var + 1]) 
            for i_var, var in enumerate(self.latents_dims.keys())
        }

    def update_prior(self):
        """
        """
        expanded = {}
        final = {}
        for parameter in self.optim_prior_parameters.keys(): # Keys should be 'mu' and 'sigma'
            # Expand by repeating (scalar into vector, matrix into block diag
            # matrix).
            expanded[parameter] = self.expand_parameters(
                self.optim_prior_parameters[parameter], 
            )

            # Concatenate the expanded parameter tensors for each latent
            # variable to form final parameters to use to instantiate
            # distribution.
            final[parameter] = self.construct_parameters(expanded[parameter])

        dist = torch.distributions.MultivariateNormal(
            loc=final['mu'], 
            scale=final['sigma']
        )
        return dist

    def update_var_post(self):
        """
        """
        dist = torch.distributions.MultivariateNormal(
            loc=self.optim_var_parameters['mu'], 
            scale=self.optim_var_parameters['sigma']
        )
        return dist

    def init_prior(self):
        """
        """
        # Original entry 'a' in self.latent_dims has (n_nodes - 2)(n_nodes - 1)
        # entries, one for each component of each acceleration vector. But we
        # want as many copies of the acceleration covariance matrix as there are
        # acceleration vectors (not total components). We also need to
        # re-instantiate the prior distribtuion at each update, so we want to
        # avoid repeating this copying step.
        self.n_copies = copy.deepcopy(self.latent_dims)
        self.n_copies['a'] = self.n_nodes - 2

        self.optim_prior_parameters = {}

        # Set as nn.Parameter. prior_init keys are 'mu' and 'sigma'.
        for parameter in self.prior_init.keys():
            self.optim_prior_parameters[parameter] = {
                var : nn.Parameter(self.prior_init[parameter][var])
                for var in self.prior_init[parameter]
            }

        self.prior = self.update_prior()
    
    def init_variational_post(self):
        """
        """
        # Dimensionality of variational distribution is sum of contribution from
        # each of the latents.
        self.var_dim = sum(self.latents_dims.values())

        if self.params.var_init == 'random':
            self.optim_var_parameters['mu'] = nn.Parameter(
                torch.abs(torch.randn(self.var_dim, dtype=self.params.precision))
            )
            self.optim_var_parameters['sigma'] = nn.Parameter(
                torch.diag(self.var_dim, dtype=self.params.precision)
            )
        else:
            raise Exception("Unrecognized value for `params.var_init`.")
        
        self.q = self.update_var_post()
            
    def expand_parameters(self, parameters):
        """
        """
        # self.n_copies was initialized in the `init_prior` method.
        expanded = {}
        for var, param in parameters.items():
            if len(param.shape) == 1:
                expanded[var] = param.repeat(self.n_copies[var])
            elif len(param.shape) == 2:
                copies = [param.clone() for _ in self.n_copies[var]]
                expanded[var] = torch.block_diag(*copies)
            else:
                raise Exception(f"Invalid shape for {var}. Must be 1D or 2D.")
            
        return expanded
    
    def construct_parameters(expanded, parameter):
        """
        
        """
        # Concatenate mu tensors to form mu vector.
        if parameter == 'mu':
            mu_tuple = tuple([tensor for tensor in expanded.values()])
            out = torch.cat(mu_tuple)
        # Put sigma on diagonal of matrix if is 1D, (if already is 2D, leave in
        # place), then concatenate to form final covariance matrix.
        elif parameter == 'sigma':
            sigma_list = [
            torch.diag(tensor) if len(tensor.shape) == 1 else tensor 
            for tensor in expanded.values()
        ]
            out = torch.block_diag(*sigma_list)
        else:
            raise Exception(f"Unrecognized value for `{parameter}`.")
        
        return out

    def sample_local_vars(self, n_samples=1):
        """
        Instantation of abstract method, draws samples from full variational
        posterior and then passes samples through transforms. Note that `a` will
        be normed and orthogonalized in trajectory construction (see
        `construct_trajectory` method).

        Parameters
        ----------
        n_samples : (int), specify number of samples to be drawn from 
                    variational posterior.

        Returns
        -------
        local_vars : (dict)
            'd' : n_samples x (n_nodes - 1) tensor of samples (after rectifying
            transform) 'c' : n_samples x (n_nodes - 2) tensor of samples (after
            identity transform) 'a' : n_samples x (n_nodes - 1) tensor of
            samples (no transform) 'l' : n_samples x (n_nodes - 1) tensor of
            samples (after nonlinear transform)
        """
        # Each value of local_vars dict should be n_samples x n_latent_dims
        # tensor, where n_latent_dims is the dimensionality of that particular
        # latent variable (e.g. for distance, this is n_nodes - 1).
        z = self.q.rsample(sample_shape=(n_samples,)).squeeze()
        local_vars = {
            var : self.transforms(z[:, self.local_vars_ind[var]], var)
            for var in self.latents_dims.keys()
        }

        # Each sample of `a` is a vector and needs to be reshaped into a matrix
        # of shape n_nodes - 1 x n_nodes - 2 (effective dim x number of
        # acceleration vectors).
        local_vars['a'] = self.reshape_a(local_vars['a'], self.n_nodes)

        return local_vars
    
        # local_vars = { 'd' : self.rectifier(z[self.local_vars_ind['d']],
        #     'softplus'), 'c' : z[self.local_vars_ind['c']], 'a' :
        #     z[self.local_vars_ind['a']], 'l' :
        #     normal_cdf(z[self.local_vars_ind['l']], mu=0, sigma2=1) }
        
    def compute_kl_divergence(self):
        """
        Instantation of abstract method, wraps
        torch.distributions.kl_divergence.
        """
        return kl_divergence(self.q, self.prior)



    


