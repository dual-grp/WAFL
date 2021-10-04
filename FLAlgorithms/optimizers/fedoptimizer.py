from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr, L_k = 0):
        defaults = dict(lr=lr, L_k = L_k)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                p.data = p.data - group['lr'] * (p.grad.data + group['L_k'] * p.data)
        return loss