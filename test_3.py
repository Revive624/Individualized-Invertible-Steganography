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


net1 = Model_3()
net2 = Model_2()
net3 = Model_2()
net1.cuda()
net2.cuda()
net3.cuda()
init_model(net1)
init_model(net2)
init_model(net3)
net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
params_trainable3 = (list(filter(lambda p: p.requires_grad, net3.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim3 = torch.optim.Adam(params_trainable3, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, c.weight_step, gamma=c.gamma)
dwt = common.DWT()
iwt = common.IWT()

if c.pretrain:
    load(c.PRETRAIN_PATH3 + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH3 + c.suffix_pretrain + '_2.pt', net2, optim2)
    load(c.PRETRAIN_PATH3 + c.suffix_pretrain + '_3.pt', net3, optim3)


with torch.no_grad():
    net1.eval()
    net2.eval()
    net3.eval()
    for i, x in enumerate(datasets.testloader):
        x = x.to(device)
        cover = x[:x.shape[0] // 4]  # channels = 3
        secret_1 = x[x.shape[0] // 4: 2 * (x.shape[0] // 4)]
        secret_2 = x[2 * (x.shape[0] // 4): 3 * (x.shape[0] // 4)]
        secret_3 = x[3 * (x.shape[0] // 4): 4 * (x.shape[0] // 4)]

        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        secret_dwt_2 = dwt(secret_2)
        secret_dwt_3 = dwt(secret_3)
        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1, secret_dwt_2, secret_dwt_3), 1)  # channels = 36

        output_dwt_1 = net1(input_dwt_1)  # channels = 72 [stego, z, z_key, global_key, local_key, key_input]
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)
        global_key_dwt = output_dwt_1.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
        local_key_dwt_1 = output_dwt_1.narrow(1, 16 * c.channels_in, 4 * c.channels_in)
        key_dwt_1 = output_dwt_1.narrow(1, 20 * c.channels_in, 4 * c.channels_in)
        output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3
        key_1 = iwt(key_dwt_1)
        z_1 = iwt(z_dwt_1)

        input_dwt_2 = torch.cat((output_steg_dwt_1, secret_dwt_2, global_key_dwt), 1)  # channels = 36

        output_dwt_2 = net2(input_dwt_2)  # channels = 48 [stego, z, z_key, local_key]
        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        local_key_dwt_2 = output_dwt_2.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
        output_steg_2 = iwt(output_steg_dwt_2)

        input_dwt_3 = torch.cat((output_steg_dwt_2, secret_dwt_3, global_key_dwt), 1)  # channels = 36

        output_dwt_3 = net3(input_dwt_3)  # channels = 48 [stego, z, z_key, local_key]
        output_steg_dwt_3 = output_dwt_3.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        local_key_dwt_3 = output_dwt_3.narrow(1, 12 * c.channels_in, 4 * c.channels_in)

        output_steg_3 = iwt(output_steg_dwt_3)  # channels = 3

        output_rev_dwt_3 = output_steg_dwt_3  # channels = 12
        rev_dwt_3 = net3(output_rev_dwt_3, rev=True)  # channels = 48 [stego, secret, key, rev_z]
        rev_steg_dwt_2 = rev_dwt_3.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        rev_secret_dwt_3 = rev_dwt_3.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_steg_2 = iwt(rev_steg_dwt_2)  # channels = 3
        rev_secret_3 = iwt(rev_secret_dwt_3)  # channels = 3

        output_rev_dwt_2 = rev_steg_dwt_2  # channels = 12
        rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 48 [stego, secret, key, rev_z]
        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
        rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

        output_rev_dwt_1 = rev_steg_dwt_1  # channels = 12
        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 60 [cover, secret, key, rev_z_key, rev_z]
        rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_secret_1 = iwt(rev_secret_dwt)

        torchvision.utils.save_image(cover, c.TEST_PATH3_cover + '%.5d.png' % i)
        torchvision.utils.save_image(secret_1, c.TEST_PATH3_secret_1 + '%.5d.png' % i)
        torchvision.utils.save_image(secret_2, c.TEST_PATH3_secret_2 + '%.5d.png' % i)
        torchvision.utils.save_image(secret_3, c.TEST_PATH3_secret_3 + '%.5d.png' % i)

        torchvision.utils.save_image(output_steg_1, c.TEST_PATH3_steg_1 + '%.5d.png' % i)
        torchvision.utils.save_image(rev_secret_1, c.TEST_PATH3_secret_rev_1 + '%.5d.png' % i)

        torchvision.utils.save_image(output_steg_2, c.TEST_PATH3_steg_2 + '%.5d.png' % i)
        torchvision.utils.save_image(rev_secret_2, c.TEST_PATH3_secret_rev_2 + '%.5d.png' % i)

        torchvision.utils.save_image(output_steg_3, c.TEST_PATH3_steg_3 + '%.5d.png' % i)
        torchvision.utils.save_image(rev_secret_3, c.TEST_PATH3_secret_rev_3 + '%.5d.png' % i)


