import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch.optim.optimizer import Optimizer

class KFAC(Optimizer):
    """
    K-FAC Preconditioner for Linear and Conv2d layers.
    This optimizer computes the Kronecker-Factored Approximate Curvature (K-FAC) 
    to precondition the gradients, enhancing the training of deep neural networks.

    Attributes:
        net (torch.nn.Module): Network to apply KFAC.
        eps (float): Tikhonov regularization parameter (epsilon).
        sua (bool): Flag to apply Structured Unit Approximation (SUA).
        pi (bool): Flag to apply pi correction for the regularization term.
        update_freq (int): Frequency of updating the preconditioner.
        alpha (float): Decay factor for running averages of the covariance matrices.
        constraint_norm (bool): Flag to scale gradients by the Fisher norm.
    """

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        super(KFAC, self).__init__(net.parameters(), {})
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        
        for mod in net.modules():
            mod_name = mod.__class__.__name__
            if mod_name in ['CRD', 'CLS']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                
                for sub_mod in mod.modules():
                    if hasattr(sub_mod, 'weight'):
                        handle = sub_mod.register_backward_hook(self._save_grad_output)
                        self._bwd_handles.append(handle)
                        
                        params = [sub_mod.weight]
                        if sub_mod.bias is not None:
                            params.append(sub_mod.bias)

                        self.param_groups.append({'params': params, 'mod': mod, 'sub_mod': sub_mod})

    def step(self, update_stats=True, update_params=True, lam=0.):
        """Perform a single optimization step."""
        self.lam = lam
        fisher_norm = 0.

        for group in self.param_groups:
            weight, bias = (group['params'] + [None])[:2]
            state = self.state[weight]

            if update_stats:
                if self._iteration_counter % self.update_freq == 0 or self.alpha != 1:
                    self._compute_covs(group, state)
                    state['ixxt'], state['iggt'] = self._inv_covs(state['xxt'], state['ggt'], state['num_locations'])

            if update_params:
                gw, gb = self._precond(weight, bias, group, state)
                fisher_norm += self._update_gradients(weight, bias, gw, gb)

        if update_params and self.constraint_norm and fisher_norm > 0:
            self._scale_gradients(fisher_norm)

        if update_stats:
            self._iteration_counter += 1

    def _update_gradients(self, weight, bias, gw, gb):
        """Update the gradients in-place and compute the norm for scaling."""
        fisher_norm = (weight.grad * gw).sum()
        weight.grad.data = gw
        if bias is not None:
            fisher_norm += (bias.grad * gb).sum()
            bias.grad.data = gb
        return fisher_norm

    def _scale_gradients(self, fisher_norm):
        """Scale the gradients by the Fisher norm."""
        scale = (1. / fisher_norm) ** 0.5
        for group in self.param_groups:
            for param in group['params']:
                param.grad.data *= scale

    # Utility functions for KFAC: _save_input, _save_grad_output, _precond, _compute_covs, _inv_covs, etc.
    # These functions manage the computation of covariances, their inverses, and the preconditioned gradients.
    # Details of these methods are crucial for understanding the internal workings of the KFAC optimizer.

    def __del__(self):
        """Clean up hooks when the optimizer is deleted."""
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()
