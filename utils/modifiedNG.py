import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ModifiedNGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False, F_modified):
        '''
        F_modified: the Fisher inverse modified by the criterion  [J, U, Lambda2, Lambda2_modified]
        '''
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        # self.params = params
        # for i,z in enumerate(self.params):
        #     print('Entered! ')
        # print('params check ?1: ', self.params)
        self.F_modified = F_modified
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, F_modified = F_modified)
        super(ModifiedNGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        has_sparse_grad = False
        # P_check = []
        for group in self.param_groups:
            for p in group['params']:     
                # print('GROUP params:   ', p.shape)           
                # P_check.append(p.reshape(-1))
                # print('param shape check: ', p.shape)
                if p.grad is not None:   
                    params_with_grad.append(p)
                    # print('\np.gradient check| max: ', torch.max(p.grad), ' | min: ', torch.min(p.grad))
                    d_p_list.append(p.grad)
                    # print('GROUP params grad shape:   ', p.grad.shape)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
        # print('step params check? : ', self.params)
        # for i,z in enumerate(self.params):
        #     print('Entered! ')
        #     p = z
        #     P_check.append(p.reshape(-1))
        #     # print('param shape check: ', p.shape)
        #     if p.grad is not None:   
        #         params_with_grad.append(p)
        #         # print('\np.gradient check| max: ', torch.max(p.grad), ' | min: ', torch.min(p.grad))
        #         d_p_list.append(p.grad)
        #         if p.grad.is_sparse:
        #             has_sparse_grad = True

        #         state = self.state[p]
        # torch.save(torch.cat(P_check), 'P_check_in_optimizer.pt')
        shape_list = [d_p_list[i].shape for i in range(len(d_p_list))]
        reshaped_d_p = torch.cat([d_p_list[i].reshape(-1,1) for i in range(len(d_p_list))], dim=0)  # shape (P,1)
        # torch.save(reshaped_d_p, 'J_L.pt')
        # print('Shape CHECK: ', self.F_modified[0].shape, self.F_modified[1].shape, 'reshaped_d_p.shape ', reshaped_d_p.shape)  # shape (P,P), (++P,), (P,1)
        # d_p_list = self.F_modified[0] @ ((self.F_modified[1] * self.F_modified[0].T) @ reshaped_d_p)
        # d_p_list = (self.F_modified[0] * self.F_modified[1]) @ (self.F_modified[0].T @ reshaped_d_p)
        # print('Computation check: ', self.F_modified[0]@self.F_modified[0].T, self.F_modified[0].T@self.F_modified[0])
        # d_p_list = torch.linalg.solve((self.F_modified[0] * self.F_modified[1]) @ (self.F_modified[0].T), reshaped_d_p)
        print('max of grad d_p: ', torch.max(reshaped_d_p))
        print('min of grad d_p: ', torch.min(reshaped_d_p))
        J, U, Lambda2, Lambda2_modified, alpha = self.F_modified
        K = J@J.T 

        Jta = J.T@alpha/J.shape[0]
        print('max|min: (J_L, Jta/N) ',f'({torch.max(reshaped_d_p)}, {torch.max(Jta)}, ratio: {torch.max(Jta)/torch.max(reshaped_d_p)})|({torch.min(reshaped_d_p)}, {torch.min(Jta)})' )
        res = reshaped_d_p.reshape(-1) - Jta.reshape(-1)
        print('\n check Jacobi res: ', res.shape, 'max: ', torch.max(res), 'mean: ', torch.mean(res), 'min: ', torch.min(res), 'norm: ', torch.linalg.norm(res), 'MSE: ', torch.linalg.norm(res)/res.shape[0])


        if torch.linalg.norm(res)/res.shape[0] >= 1e-6:
            print('BAD Jacobian OCCURS!')
            # learningrate = group['lr']
            # d_p_list = reshaped_d_p
        test_gradient = reshaped_d_p
        test_solve = torch.linalg.solve(torch.matmul(K,K), torch.matmul(J, test_gradient))
        test_solve = torch.matmul(J.T, test_solve)
        # print('Shape check: ', torch.matmul(J.T, torch.matmul(J, test_solve)).shape, test_gradient.shape)
        res = torch.abs(torch.matmul(J.T, torch.matmul(J, test_solve)) - test_gradient)
        print('\n check NTK dimension reduction res: ', res.shape, 'max: ', torch.max(res), 'mean: ', torch.mean(res), 'min: ', torch.min(res), 'norm: ', torch.linalg.norm(res), 'MSE: ', torch.linalg.norm(res)/res.shape[0])
        sigma02 = 1e-2
        d_p_list = torch.linalg.solve(K@(U@torch.diag_embed(0.9*Lambda2_modified/Lambda2+0.1)@U.T)@K, J@reshaped_d_p*sigma02*J.shape[0])
        # d_p_list = torch.linalg.solve((self.F_modified[0]@self.F_modified[0].T) @ (self.F_modified[0]@self.F_modified[0].T) + (torch.rand(1, device=self.F_modified[0].device) *0.9 + 0.1) *self.F_modified[0]@self.F_modified[0].T, self.F_modified[0]@reshaped_d_p*self.F_modified[1])
        d_p_list = (J.T)@d_p_list
        learningrate = group['lr']

        print('Shape check: ', d_p_list.shape)
        print('max of d_p_list: ', torch.max(d_p_list)) 
        print('min of d_p_list: ', torch.min(d_p_list))
        # d_p_list = torch.linalg.solve((self.F_modified.T) @ (self.F_modified), reshaped_d_p)

        # print('d_p_list shape check: ', d_p_list.shape)
        # print('\nupdate check| max: ', torch.max(d_p_list), ' | min: ', torch.min(d_p_list))
        len_list = []
        for i in range(len(shape_list)):
            l = 1
            for u in shape_list[i]:
                l *= u
            len_list.append(l)
        # print('len_LIST', len_list)
        d_p_list = [d_p_list[sum(len_list[:i]): sum(len_list[:i+1])].reshape(shape_list[i]) for i in range(len(shape_list))]
        # print('hhhhhh', [d_p_list[i].shape for i in range(len(d_p_list))])
        modifiedngd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=learningrate,
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

        return loss


def modifiedngd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs modified NGD algorithm computation.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_ngd
    else:
        func = _single_tensor_ngd

    func(params,
         d_p_list,
         momentum_buffer_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_ngd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


def _multi_tensor_ngd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
