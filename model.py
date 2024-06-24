import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet_stage1
from prm import PredictiveModule
from denseblock import Dense


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


class Model_0(nn.Module):
    def __init__(self, channel_in=12, nf=12):
        super(Model_0, self).__init__()

        self.model = Hinet_stage1()
        self.prm = PredictiveModule(channel_in, nf)
        self.channel = channel_in
        self.local_key_create = Dense(2*channel_in, channel_in)
        self.rev_local_key_create = Dense(channel_in, channel_in)

    def forward(self, x, rev=False):

        if not rev:
            cover = x[:, 0:self.channel]
            secret_1 = x[:, self.channel:self.channel*2]
            local_key = self.local_key_create(torch.cat((cover, secret_1), 1))
            key_enc = c.lamda_part_key * local_key
            x = torch.cat((cover, secret_1, key_enc), 1)
            out = self.model(x)
            out = torch.cat((out, local_key, key_enc), 1)  # channels = 60

        else:
            stego = x
            local_key = self.rev_local_key_create(stego)
            z = self.prm(stego)
            key_enc = c.lamda_part_key * local_key
            x = torch.cat((stego, z, key_enc), 1)
            out = self.model(x, rev=True)
            out = torch.cat((out, z), 1)  # channels = 48

        return out


class Model_1(nn.Module):
    def __init__(self, channel_in=12, nf=12):
        super(Model_1, self).__init__()

        self.model = Hinet_stage1()
        self.prm = PredictiveModule(channel_in, nf)
        self.channel = channel_in
        self.global_key_create = Dense(3*channel_in, channel_in)
        self.local_key_create = Dense(2*channel_in, channel_in)
        self.rev_global_key_create = Dense(channel_in, channel_in)
        self.rev_local_key_create = Dense(channel_in, channel_in)

    def forward(self, x, rev=False):

        if not rev:
            cover = x[:, 0:self.channel]
            secret_1 = x[:, self.channel:self.channel*2]
            secret_2 = x[:, self.channel*2:self.channel*3]
            global_key = self.global_key_create(torch.cat((cover, secret_1, secret_2), 1))
            local_key = self.local_key_create(torch.cat((cover, secret_1), 1))
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((cover, secret_1, key_enc), 1)
            out = self.model(x)
            out = torch.cat((out, global_key, local_key, key_enc), 1)  # channels = 72

        else:
            stego = x
            global_key = self.rev_global_key_create(stego)
            local_key = self.rev_local_key_create(stego)
            z = self.prm(stego)
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((stego, z, key_enc), 1)
            out = self.model(x, rev=True)
            out = torch.cat((out, z), 1)  # channels = 48

        return out


class Model_2(nn.Module):
    def __init__(self, channel_in=12, nf=12):
        super(Model_2, self).__init__()

        self.model = Hinet_stage1()
        self.prm = PredictiveModule(channel_in, nf)
        self.channel = channel_in
        self.local_key_create = Dense(2*channel_in, channel_in)
        self.rev_global_key_create = Dense(channel_in, channel_in)
        self.rev_local_key_create = Dense(channel_in, channel_in)

    def forward(self, x, rev=False):

        if not rev:
            cover = x[:, 0:self.channel]
            secret = x[:, self.channel*1:self.channel*2]
            global_key = x[:, self.channel*2:self.channel*3]
            local_key = self.local_key_create(torch.cat((cover, secret), 1))
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((cover, secret, key_enc), 1)
            out = self.model(x)
            out = torch.cat((out, local_key), 1)  # channels = 48

        else:
            stego = x
            global_key = self.rev_global_key_create(stego)
            local_key = self.rev_local_key_create(stego)
            z = self.prm(stego)
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((stego, z, key_enc), 1)
            out = self.model(x, rev=True)
            out = torch.cat((out, z), 1)  # channels = 48

        return out


class Model_3(nn.Module):
    def __init__(self, channel_in=12, nf=12):
        super(Model_3, self).__init__()

        self.model = Hinet_stage1()
        self.prm = PredictiveModule(channel_in, nf)
        self.channel = channel_in
        self.global_key_create = Dense(4*channel_in, channel_in)
        self.local_key_create = Dense(2*channel_in, channel_in)
        self.rev_global_key_create = Dense(channel_in, channel_in)
        self.rev_local_key_create = Dense(channel_in, channel_in)

    def forward(self, x, rev=False):

        if not rev:
            cover = x[:, 0:self.channel]
            secret_1 = x[:, self.channel:self.channel*2]
            secret_2 = x[:, self.channel*2:self.channel*3]
            secret_3 = x[:, self.channel*3:self.channel*4]
            global_key = self.global_key_create(torch.cat((cover, secret_1, secret_2, secret_3), 1))
            local_key = self.local_key_create(torch.cat((cover, secret_1), 1))
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((cover, secret_1, key_enc), 1)
            out = self.model(x)
            out = torch.cat((out, global_key, local_key, key_enc), 1)  # channels = 72

        else:
            stego = x
            global_key = self.rev_global_key_create(stego)
            local_key = self.rev_local_key_create(stego)
            z = self.prm(stego)
            key_enc = c.lamda_overall_key * global_key + c.lamda_part_key * local_key
            x = torch.cat((stego, z, key_enc), 1)
            out = self.model(x, rev=True)
            out = torch.cat((out, z), 1)  # channels = 48

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
