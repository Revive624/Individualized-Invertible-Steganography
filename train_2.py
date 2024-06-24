import torch.nn
import torch.optim
import math
import numpy as np
from model import *
from imp_subnet import *
import config as c
from datasets import trainloader, valloader
import modules.Unet_common as common
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)

    if mse < 1.0e-10:
        return 100

    if mse > 1.0e15:
        return -100

    return 10 * math.log10(255.0 ** 2 / mse)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def l2_loss(target, original):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(target, original)
    return loss.to(device)


def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.to(device)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).cuda()


net1 = Model_1()
net2 = Model_2()
net1.cuda()
net2.cuda()
init_model(net1)
init_model(net2)
net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
para1 = get_parameter_number(net1)
para2 = get_parameter_number(net2)
print(para1)
print(para2)
params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.train_next:
    load(c.MODEL_PATH2 + c.suffix_load + '_1.pt', net1, optim1)
    load(c.MODEL_PATH2 + c.suffix_load + '_2.pt', net2, optim2)

if c.pretrain:
    load(c.PRETRAIN_PATH2 + c.suffix_pretrain + '_1.pt', net1, optim1)
    load(c.PRETRAIN_PATH2 + c.suffix_pretrain + '_2.pt', net2, optim2)

for i_epoch in range(c.epochs):
    i_epoch = i_epoch + c.trained_epoch + 1
    loss_history = []
    loss_history_g1 = []
    loss_history_g2 = []
    loss_history_r1 = []
    loss_history_r2 = []
    loss_history_z1 = []
    loss_history_z2 = []
    net1.train()
    net2.train()
    for i_batch, data in enumerate(trainloader):
        data = data.to(device)
        cover = data[:data.shape[0] // 3]  # channels = 3
        secret_1 = data[data.shape[0] // 3: 2 * (data.shape[0] // 3)]
        secret_2 = data[2 * (data.shape[0] // 3): 3 * (data.shape[0] // 3)]
        cover_dwt = dwt(cover)  # channels = 12
        secret_dwt_1 = dwt(secret_1)
        secret_dwt_2 = dwt(secret_2)
        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1, secret_dwt_2), 1)  # channels = 36

        output_dwt_1 = net1(input_dwt_1)  # channels = 72 [stego, z, z_key, global_key, local_key, key_input]
        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        global_key_dwt = output_dwt_1.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
        local_key_dwt_1 = output_dwt_1.narrow(1, 16 * c.channels_in, 4 * c.channels_in)
        key_input_dwt_1 = output_dwt_1.narrow(1, 20 * c.channels_in, 4 * c.channels_in)
        output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3

        input_dwt_2 = torch.cat((output_steg_dwt_1, secret_dwt_2, global_key_dwt), 1)  # channels = 48

        output_dwt_2 = net2(input_dwt_2)  # channels = 48 [stego, z, z_key, local_key]
        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 24
        local_key_dwt_2 = output_dwt_2.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
        output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

        output_rev_dwt_2 = output_steg_dwt_2  # channels = 12
        rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 48 [stego, secret, key, rev_z]
        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
        rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_z_dwt_2 = rev_dwt_2.narrow(1, 12 * c.channels_in, 4 * c.channels_in)  # channels = 12

        rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
        rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

        output_rev_dwt_1 = rev_steg_dwt_1  # channels = 12
        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 48 [cover, secret, key, rev_z]
        rev_secret_dwt_1 = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
        rev_z_dwt_1 = rev_dwt_1.narrow(1, 12 * c.channels_in, 4 * c.channels_in)

        rev_secret_1 = iwt(rev_secret_dwt_1)

        g_loss_1 = l2_loss(output_steg_1.cuda(), cover.cuda())
        g_loss_2 = l2_loss(output_steg_2.cuda(), cover.cuda())

        r_loss_1 = l2_loss(rev_secret_1.cuda(), secret_1.cuda())
        r_loss_2 = l2_loss(rev_secret_2.cuda(), secret_2.cuda())

        z_loss_1 = l2_loss(rev_z_dwt_1.cuda(), output_z_dwt_1.cuda())
        z_loss_2 = l2_loss(rev_z_dwt_2.cuda(), output_z_dwt_2.cuda())

        total_loss = c.lamda_reconstruction_1 * r_loss_1 + c.lamda_reconstruction_2 * r_loss_2 + \
                     c.lamda_guide_1 * g_loss_1 + c.lamda_guide_2 * g_loss_2
        total_loss.backward()

        optim1.step()
        optim2.step()

        optim1.zero_grad()
        optim2.zero_grad()

        loss_history.append(total_loss.item())
        loss_history_g1.append(g_loss_1.item())
        loss_history_g2.append(g_loss_2.item())
        loss_history_r1.append(r_loss_1.item())
        loss_history_r2.append(r_loss_2.item())
        loss_history_z1.append(z_loss_1.item())
        loss_history_z2.append(z_loss_2.item())

    epoch_losses = np.mean(np.array(loss_history))
    epoch_losses_g1 = np.mean(np.array(loss_history_g1))
    epoch_losses_g2 = np.mean(np.array(loss_history_g2))
    epoch_losses_r1 = np.mean(np.array(loss_history_r1))
    epoch_losses_r2 = np.mean(np.array(loss_history_r2))
    epoch_losses_z1 = np.mean(np.array(loss_history_z1))
    epoch_losses_z2 = np.mean(np.array(loss_history_z2))
    print('[%d/%d] g1:%.8f\tg2:%.8f\tr1:%.8f\tr2:%.8f\nz1:%.8f\tz2:%.8f\ttotal:%.8f\n' %
          (i_epoch, c.epochs, epoch_losses_g1.item(), epoch_losses_g2.item(), epoch_losses_r1.item(),
           epoch_losses_r2.item(), epoch_losses_z1.item(), epoch_losses_z2.item(), epoch_losses.item()))

    if c.single_device and i_epoch % c.val_freq == 0:
        with torch.no_grad():
            psnr_s1 = []
            psnr_s2 = []
            psnr_c1 = []
            psnr_c2 = []
            net1.eval()
            net2.eval()
            for x in valloader:
                x = x.to(device)
                cover = x[:x.shape[0] // 3]  # channels = 3
                secret_1 = x[x.shape[0] // 3: 2 * (x.shape[0] // 3)]
                secret_2 = x[2 * (x.shape[0] // 3): 3 * (x.shape[0] // 3)]

                cover_dwt = dwt(cover)  # channels = 12
                secret_dwt_1 = dwt(secret_1)
                secret_dwt_2 = dwt(secret_2)
                input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1, secret_dwt_2), 1)  # channels = 36

                output_dwt_1 = net1(input_dwt_1)  # channels = 72 [stego, z, z_key, global_key, local_key, key_input]
                output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                global_key_dwt = output_dwt_1.narrow(1, 12 * c.channels_in, 4 * c.channels_in)
                local_key_dwt_1 = output_dwt_1.narrow(1, 16 * c.channels_in, 4 * c.channels_in)
                output_steg_1 = iwt(output_steg_dwt_1)  # channels = 3

                input_dwt_2 = torch.cat((output_steg_dwt_1, secret_dwt_2, global_key_dwt), 1)  # channels = 36

                output_dwt_2 = net2(input_dwt_2)  # channels = 48 [stego, z, z_key, local_key]
                output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                local_key_dwt_2 = output_dwt_2.narrow(1, 12 * c.channels_in, 4 * c.channels_in)

                output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3

                output_rev_dwt_2 = output_steg_dwt_2  # channels = 12
                rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 48 [stego, secret, key, rev_z]
                rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                rev_steg_1 = iwt(rev_steg_dwt_1)  # channels = 3
                rev_secret_2 = iwt(rev_secret_dwt_2)  # channels = 3

                output_rev_dwt_1 = rev_steg_dwt_1  # channels = 12
                rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 48 [cover, secret, key, rev_z]
                rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                rev_secret_1 = iwt(rev_secret_dwt)

                secret_rev1_255 = rev_secret_1.cpu().numpy().squeeze() * 255
                secret_rev2_255 = rev_secret_2.cpu().numpy().squeeze() * 255
                secret_1_255 = secret_1.cpu().numpy().squeeze() * 255
                secret_2_255 = secret_2.cpu().numpy().squeeze() * 255

                cover_255 = cover.cpu().numpy().squeeze() * 255
                steg_1_255 = output_steg_1.cpu().numpy().squeeze() * 255
                steg_2_255 = output_steg_2.cpu().numpy().squeeze() * 255

                psnr_temp1 = computePSNR(secret_rev1_255, secret_1_255)
                psnr_s1.append(psnr_temp1)
                psnr_temp2 = computePSNR(secret_rev2_255, secret_2_255)
                psnr_s2.append(psnr_temp2)

                psnr_temp_c1 = computePSNR(cover_255, steg_1_255)
                psnr_c1.append(psnr_temp_c1)
                psnr_temp_c2 = computePSNR(cover_255, steg_2_255)
                psnr_c2.append(psnr_temp_c2)

            print('validation:  psnr_c1:%.4f\tpsnr_c2:%.4f\tpsnr_s1:%.4f\tpsnr_s2:%.4f\n' %
                  (np.mean(np.array(psnr_c1)).item(), np.mean(np.array(psnr_c2)).item(),
                   np.mean(np.array(psnr_s1)).item(), np.mean(np.array(psnr_s2)).item()))

    if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH2 + 'model_checkpoint_%.5i_1' % i_epoch + '.pt')
        torch.save({'opt': optim2.state_dict(),
                    'net': net2.state_dict()}, c.MODEL_PATH2 + 'model_checkpoint_%.5i_2' % i_epoch + '.pt')
    weight_scheduler1.step()
    weight_scheduler2.step()

torch.save({'opt': optim1.state_dict(),
            'net': net1.state_dict()}, c.MODEL_PATH2 + 'model_1' + '.pt')
torch.save({'opt': optim2.state_dict(),
            'net': net2.state_dict()}, c.MODEL_PATH2 + 'model_2' + '.pt')
