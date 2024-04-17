import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch.optim.optimizer import Optimizer

class KFAC(Optimizer):
    """
    K-FAC Preconditioner for Linear and Conv2d layers.
    This optimizer computes the Kronecker-Factored Approximate Curvature (K-FAC) 
    to precondition the gradients, enhancing the training of deep neural networks.
    """
    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """
        Initializes the KFAC optimizer.
        
        Parameters:
            net (torch.nn.Module): Network to apply KFAC.
            eps (float): Tikhonov regularization parameter (epsilon).
            sua (bool): Flag to apply Structured Unit Approximation (SUA).
            pi (bool): Flag to apply pi correction for the regularization term.
            update_freq (int): Frequency of updating the preconditioner.
            alpha (float): Decay factor for running averages of the covariance matrices.
            constraint_norm (bool): Flag to scale gradients by the Fisher norm.
        """
        super(KFAC, self).__init__(net.parameters(), {})
        self.eps = eps  # Regularization parameter
        self.sua = sua  # Structured Unit Approximation flag
        self.pi = pi    # Pi correction flag
        self.update_freq = update_freq  # Update frequency for the preconditioner
        self.alpha = alpha  # Decay factor for covariance matrices
        self.constraint_norm = constraint_norm  # Fisher norm scaling flag
        self.params = []  # List to hold parameters
        self._fwd_handles = []  # List to hold forward hooks
        self._bwd_handles = []  # List to hold backward hooks
        self._iteration_counter = 0  # Counter for preconditioner updates
        
        # Register hooks to save inputs and gradients
        for mod in net.modules():
            mod_name = mod.__class__.__name__
            if mod_name in ['CRD', 'CLS']:  # Check if module is a target for KFAC
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                
                for sub_mod in mod.modules():
                    if hasattr(sub_mod, 'weight'):
                        handle = sub_mod.register_backward_hook(self._save_grad_output)
                        self._bwd_handles.append(handle)
                        
                        params = [sub_mod.weight]  # Parameters to precondition
                        if sub_mod.bias is not None:
                            params.append(sub_mod.bias)

                        self.param_groups.append({'params': params, 'mod': mod, 'sub_mod': sub_mod})

    def step(self, update_stats=True, update_params=True, lam=0.):
        """
        Perform a single optimization step.

        Parameters:
            update_stats (bool): Whether to update running statistics.
            update_params (bool): Whether to update parameters.
            lam (float): Lambda for additional regularization.
        """
        self.lam = lam  # Set regularization lambda
        fisher_norm = 0.  # Initialize Fisher norm

        for group in self.param_groups:
            weight, bias = (group['params'] + [None])[:2]  # Unpack weight and bias
            state = self.state[weight]  # Get state for this group

            if update_stats:
                if self._iteration_counter % self.update_freq == 0 or self.alpha != 1:
                    self._compute_covs(group, state)  # Compute covariances
                    state['ixxt'], state['iggt'] = self._inv_covs(state['xxt'], state['ggt'], state['num_locations'])

            if update_params:
                gw, gb = self._precond(weight, bias, group, state)  # Precondition gradients
                fisher_norm += self._update_gradients(weight, bias, gw, gb)  # Update gradients

        if update_params and self.constraint_norm and fisher_norm > 0:
            self._scale_gradients(fisher_norm)  # Scale gradients by Fisher norm

        if update_stats:
            self._iteration_counter += 1  # Increment update counter

    def _update_gradients(self, weight, bias, gw, gb):
        """
        Update gradients in-place and compute the norm for scaling.

        Parameters:
            weight (Tensor): Weight parameter.
            bias (Tensor): Bias parameter (optional).
            gw (Tensor): Preconditioned gradient for weight.
            gb (Tensor): Preconditioned gradient for bias (optional).
        """
        fisher_norm = (weight.grad * gw).sum()  # Compute dot product for Fisher norm
        weight.grad.data = gw  # Update weight gradient
        if bias is not None:
            fisher_norm += (bias.grad * gb).sum()  # Update bias gradient if exists
            bias.grad.data = gb
        return fisher_norm

    def _scale_gradients(self, fisher_norm):
        """
        Scale gradients by the Fisher norm.
        """
        scale = (1. / fisher_norm) ** 0.5  # Compute scaling factor
        for group in self.param_groups:
            for param in group['params']:
                param.grad.data *= scale  # Scale gradients

    def __del__(self):
        """
        Clean up hooks when the optimizer is deleted.
        """
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()  # Remove all registered hooks
