import torch
import torch.nn as nn
from denseblock import Dense
import config as c


class INV_block_affine(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=c.clamp, harr=True, in_1=3, in_2=3):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp

        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.h = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
        # ψ
        self.p = subnet_constructor(self.split_len2, self.split_len1)
        # γ
        self.c = subnet_constructor(self.split_len2, self.split_len2)
        # δ
        self.d = subnet_constructor(self.split_len2, self.split_len2)
        # λ
        self.L = subnet_constructor(self.split_len1, self.split_len2)
        # σ
        self.s = subnet_constructor(self.split_len2, self.split_len2)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2, x3 = (x.narrow(1, 0, self.split_len1),
                      x.narrow(1, self.split_len1, self.split_len2),
                      x.narrow(1, self.split_len1 + self.split_len2, self.split_len2))

        if not rev:
            y1 = x1 * self.e(self.p(x2)) + self.f(x2) + self.c(x3)
            y2 = x2 * self.e(self.r(y1)) + self.h(y1) + self.d(x3)
            y3 = x3 + self.L(y1) + self.s(y2)

        else:  # names of x and y are swapped!
            y3 = x3 - self.L(x1) - self.s(x2)
            y2 = (x2 - self.h(x1) - self.d(y3)) / self.e(self.r(x1))
            y1 = (x1 - self.f(y2) - self.c(y3)) / self.e(self.p(y2))

        return torch.cat((y1, y2, y3), 1)
