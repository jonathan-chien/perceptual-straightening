import torch

from .axb import FreeEnergyA, FreeEnergyB
from ..utils.params import Params, set_params


def infer_global_curvature(data, params_):

    default_params = {
        'max_n_epochs' : 100,
        'lr' : 0.001,
        'accum_gradients' : True
    }

    params = set_params(default_params, params)
    
    if params.model == 'A':
        elbo = FreeEnergyA(data, params.var_params)
    else:
        raise Exception("Unrecognized value for `self.params.model`.")
    
    optimizer = torch.optim.Adam(
        elbo.parameters(), lr=params.lr
    )
    
    for _ in params.n_epochs:
        elbo.compute_loss()
        elbo.backward()
        if params.model == 'A' and not params.accum_gradients: 
            elbo = adjust_prior_gradients(elbo)
        optimizer.step()
        elbo.update()

    return elbo.optim_prior_parameters, elbo


def adjust_prior_gradients(elbo):
    """
    """
    for parameter in ['mu', 'sigma']:
        for var in ['d', 'c', 'a', 'l']:
            rescaled = (
                elbo.optim_prior_parameters[parameter][var] / elbo.n_copies[var]
            )
            with torch.no_grad:
                elbo.optim_prior_parameters[parameter][var].grad = rescaled

    return elbo


def bootstrap(data, boot_params_, inf_params_):
    default_boot_params = {
         
    }

    boot_params = set_params(default_boot_params, boot_params_,)

    results = {
        'theta' : [],
        'elbo' : []
    }
    for _ in range(boot_params.n_bootstraps):
            bootstrapped = gen_bootstrap_samples(data, boot_params.method)
            theta, elbo = infer_global_curvature(bootstrapped, inf_params_)
            results['theta'].append(theta)
            results['elbo'].append(elbo)
        
def gen_bootstrap_samples(data, method):
    """
    """
    bootstrapped = {}
    
    if method == 'non_param':
        binom = torch.distributions.Binomial(
            total_count=data.all, 
            probs=data.p_correct
        )
        bootstrapped['all'] = torch.deepcopy(data.all)
        bootstrapped['correct'] = binom.sample()
        bootstrapped['incorrect'] = bootstrapped['all'] - boostrapped['correct']
        bootstrapped = Params(bootstrapped)
    else:
        raise Exception(
            f"Unrecognized value for `self.params.boot_method`: {method}."
        )

    return bootstrapped


# class Inference:
#     """
#     """

#     default_params = {
#         'max_n_epochs' : 100,
#         'lr' : 0.001,
#         'accum_gradients' : True
#     }

#     def __init__(self, data, params, kwargs_lock=True, **kwargs):
#         """
#         """
#         self.params = set_params(
#             self.default_params,
#             params,
#             kwargs_lock=kwargs_lock,
#             **kwargs
#         )
        
#         if self.params.model == 'A':
#             self.elbo = FreeEnergyA(data, self.params.variational_params)
#         else:
#             raise Exception("Unrecognized value for `self.params.model`.")

#     def adjust_prior_gradients(self):
#         """
#         """
#         for parameter in ['mu', 'sigma']:
#             for var in ['d', 'c', 'a', 'l']:
#                 rescaled = (
#                     self.elbo.optim_prior_parameters[parameter][var] 
#                     / self.elbo.n_copies[var]
#                 )
#                 with torch.no_grad:
#                     self.elbo.optim_prior_parameters[parameter][var].grad = rescaled

#         # with torch.no_grad():
#         #     for var in local_vars:
#         #         self.prior.mu.grad[self.elbo.local_var_ind[var]] = avg_par_derivs['mu'][var]
#         #         self.prior.sigma.grad[]
        
#     def infer_global_curvature(self):
#         """
#         """
#         self.optimizer = torch.optim.Adam(
#             self.elbo.parameters(), lr=self.params.lr
#         )
        
#         for _ in self.params.n_epochs:
#             self.elbo.compute_loss()
#             self.elbo.backward()
#             if self.params.model == 'A' and not self.params.accum_gradients: 
#                 self.adjust_gradients()
#             self.optimizer.step()
#             self.elbo.update()

        
# class Bootstrapper(Inference):
#     default_params = {
#     }

#     def __init__(self, data, params, kwargs_lock=True, **kwargs):
#         super().__init__(data, params, kwargs_lock=kwargs_lock, **kwargs)

#     def gen_bootstrap_samples(self):
#         bootstrapped = {}
#         if self.params.boot_method == 'non_param':
#             binom = torch.distributions.Binomial(
#                 total_count=self.data.all, 
#                 probs=self.data.p_correct
#             )
#             boostrapped = binom.sample()
#         else:
#             raise Exception(
#                 "Unrecognized value for `self.params.boot_method`: "
#                 f"{self.params.boot_method}."
#             )

#         return bootstrapped
    
#     def bootstrap(self, parallel=False):
#         # Need to check if params has parallel key.

#         for i_bootstrap in range(n_bootstraps):
#             bootstrapped = self.gen_bootstrap_samples()
#             boot_results[idx] = inference_logic(bootstrapped)

#         self.boot_results
    
# class Bootstrapper():
#     default_params = {
#     }

#     def __init__(self, data, params, kwargs_lock=kwargs_lock, **kwargs):
#         self.params = set_params(
#             self.default_params,
#             params,
#             kwargs_lock=kwargs_lock,
#             **kwargs
#         )
#         self.data = data

#     def gen_bootstrap_samples(self):
#         if self.params.boot_method == 'non_param':
#             pass
#         else:
#             pass

#         return bootstrapped
    
#     def bootstrap(self, parallel=False):
#         # Need to check if params has parallel key.

#         for i_bootstrap in range(n_bootstraps):
#             bootstrapped = self.gen_bootstrap_samples()
#             boot_results[idx] = inference_logic(bootstrapped)

#         self.boot_results

        

    

