import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class StableSqrt(torch.autograd.Function):
    """
    Workaround to avoid the derivative of sqrt(0)
    This method returns sqrt(x) in its forward pass and in the backward pass it returns the gradient of sqrt(x) for
    all cases except for sqrt(0) where it returns the gradient 0
    """
    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors[0]
        grad = grad_output / (2.0 * result)
        grad[result == 0] = 0

        return grad


class DiscreteLayer(nn.Module, ABC):
    def __init__(self, size, binary=False):
        super().__init__()

        self.b = nn.Parameter(torch.empty(size))

        nn.init.xavier_uniform_(self.b)

        if not binary:
            self.a = nn.Parameter(torch.empty(size))
            nn.init.xavier_uniform_(self.a)

        self.size = size
        self.binary = binary

    def mean(self):
        if self.binary:
            sa = 0
        else:
            sa = torch.sigmoid(self.a)
        sb = torch.sigmoid(self.b)

        return (sa - 1) * (1 - 2 * sb)

    def var(self):
        if self.binary:
            sa = 0
        else:
            sa = torch.sigmoid(self.a)

        sb = torch.sigmoid(self.b)
        m = self.mean()

        x = 1 + m * m - 2 * m
        y = 1 + m * m + 2 * m

        xa = sb - sa * sb
        xb = 1 - sa - sb + sa * sb

        return x * xa + y * xb

    def entropy(self):
        if self.binary:
            sa = 0
            p0 = 0

        else:
            sa = torch.sigmoid(self.a)
            p0 = sa * torch.log(sa)

        sb = torch.sigmoid(self.b)

        p1 = (1 - sa) * sb
        p1 = p1 * torch.log(p1)

        p2 = (1 - sa) * (1 - sb)
        p2 = p2 * torch.log(p2)

        return -(p0 + p1 + p2)

    def get_probs(self):
        sb = torch.sigmoid(self.b)
        if self.binary:
            return [1 - sb, sb]
        else:
            sa = torch.sigmoid(self.a)
            return [(1 - sb) * (1 - sa), sa, sb * (1 - sa)]

    def sample(self):
        rnd = torch.randn(self.size, device=self.b.device)

        result = torch.empty(self.size, dtype=torch.int, device=self.b.device).fill_(-1)
        result[rnd < torch.sigmoid(self.b)] = 1

        if not self.binary:
            rnd = torch.randn(self.size, device=self.b.device)
            result[rnd < torch.sigmoid(self.a)] = 0

        return result

    @abstractmethod
    def to_discrete(self):
        pass


class DiscreteConv2d(DiscreteLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, binary=False):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        size = [out_channels, in_channels, *kernel_size]
        super().__init__(size, binary)

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        m = self.mean()
        v = self.var()

        m = F.conv2d(x, m, stride=self.stride, padding=self.padding)
        v = F.conv2d(x.pow(2), v, stride=self.stride, padding=self.padding)

        v = StableSqrt.apply(v)

        eps = torch.empty(v.size(), device=x.device).normal_(0.0, 1.0)

        return m + v * eps

    def to_discrete(self):
        result = nn.Conv2d(self.size[1], self.size[0], [self.size[2], self.size[3]],
                           self.stride, self.padding, bias=False).cuda()

        result.weight.data = self.sample().float()

        return result


class DiscreteLinear(DiscreteLayer):
    def __init__(self, in_features, out_features, binary=False):
        size = [out_features, in_features]
        super().__init__(size, binary)

    def forward(self, x):
        m = self.mean()
        v = self.var()

        m = F.linear(x, m)
        v = F.linear(x.pow(2), v)

        v = StableSqrt.apply(v)

        eps = torch.empty(v.size(), device=x.device).normal_(0.0, 1.0)

        return m + v * eps

    def to_discrete(self):
        result = nn.Linear(self.size[1], self.size[0], bias=False).cuda()

        result.weight.data = self.sample().float()

        return result

