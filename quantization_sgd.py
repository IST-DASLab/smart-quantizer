import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

from quantization import SmartQuantizer, StandardQuantizer, HadamardQuantizer, SanityQuantizer, ExponentialQuantizer


class QuantizedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params,
                 quantizer, quantization_level, bucket_size,
                 lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        self.quantizer = None
        if quantizer == "smart":
            self.quantizer = SmartQuantizer(quantization_level, bucket_size)
        elif quantizer == "standard":
            self.quantizer = StandardQuantizer(quantization_level, bucket_size)
        elif quantizer == "hadamard":
            self.quantizer = HadamardQuantizer(quantization_level, bucket_size)
        elif quantizer == "exponential":
            self.quantizer = ExponentialQuantizer(quantization_level, bucket_size)
        elif quantizer == "sanity":
            self.quantizer = SanityQuantizer(quantization_level, bucket_size)
        else:
            raise RuntimeError("There is no such quantizer %s" % quantizer)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(QuantizedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QuantizedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        whole_grad_list = None
        # whole_grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                flatten = p.grad.data.view(-1)
                if whole_grad_list is None:
                    whole_grad_list = flatten
                else:
                    whole_grad_list = torch.cat((whole_grad_list, flatten), 0)

        whole_grad_list = self.quantizer.quantize(whole_grad_list.tolist())

        id = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                old_shape = d_p.shape
                # print(d_p)
                reshaped_d_p = d_p.view(-1).tolist()
                #d_p = torch.FloatTensor(self.quantizer.quantize(reshaped_d_p))
                d_p = torch.FloatTensor(whole_grad_list[id:id + len(reshaped_d_p)])
                d_p.resize_(old_shape)
                d_p = d_p.to(p.grad.get_device())

                id += len(reshaped_d_p)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
