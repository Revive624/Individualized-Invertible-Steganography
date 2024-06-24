import math
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
from imp_subnet import *
import config as c
import datasets
import modules.Unet_common as common

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


net1 = Model_0()
net1.cuda()
init_model(net1)
net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
dwt = common.DWT()
iwt = common.IWT()

if c.pretrain:
    load(c.PRETRAIN_PATH1 + c.suffix_pretrain + '_1.pt', net1, optim1)

with torch.no_grad():
    net1.eval()
    for i, x in enumerate(datasets.testloader):
        x = x.to(device)
        cover = x[:x.shape[0] // 2]  # channels = 3
        secret_1 = x[x.shape[0] // 2: 2 * (x.shape[0] // 2)]

        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24

        output_dwt_1 = net1(input_dwt_1)  # channels = 60 [stego, z, z_key, local_key, key_input]
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)
        local_key_dwt_1 = output_dwt_1.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
        key_dwt_1 = output_dwt_1.narrow(1, 16 * c.channels_in, 4 * c.channels_in)

        output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3
        key_1 = iwt(key_dwt_1)
        z_1 = iwt(z_dwt_1)

        output_rev_dwt_1 = output_steg_dwt_1  # channels = 12
        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 48 [cover, secret, key, rev_z]
        rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt)

        torchvision.utils.save_image(cover, c.TEST_PATH1_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret_1, c.TEST_PATH1_secret_1 + '%.5d.png' % i)

        torchvision.utils.save_image(output_steg_1, c.TEST_PATH1_steg_1 + '%.5d.png' % i)
        torchvision.utils.save_image(rev_secret_1, c.TEST_PATH1_secret_rev_1 + '%.5d.png' % i)

