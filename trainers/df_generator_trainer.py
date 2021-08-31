import torch
from torch.autograd import Variable
import torchvision.models as models
import time
from models.unet import UNet
from models.dafl_generator import Generator32, Discriminator, Generator256, Generator128
from models.cifar10_dcgan_generator_discriminator import Generator_cifar10, Discriminator_cifar10, weights_init
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from utils.general import get_generator_initial_noise
import random
from tqdm import tqdm
import os
from losses import StyleLoss, DeepInversionFeatureHook
import torch.nn.functional as nnf

def get_generator(args):
    if args.generator_type == 'generator32':
        generator = Generator32(args)
    elif args.generator_type == 'generator128':
        generator = Generator128(args)
    elif args.generator_type == 'generator256':
        generator = Generator256(args)
    elif args.generator_type == 'unet':
        generator = UNet(num_input_channels=3, num_output_channels=3,
                   feature_scale=8, more_layers=1,
                   concat_x=False, upsample_mode='deconv',
                   pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)
    return generator

def train_df_generator(teacher_model, task_id, args):
    if args.load_G and task_id == args.g_resume_iteration:
        generator_path = args.teacher_model_path + args.load_G_name
        generator = torch.load(generator_path).cuda()
        print(generator_path)
    elif task_id > args.g_resume_iteration:
        ### set tensorboard path
        tensorboard_G_path = args.tensorboard_base_path + 'task{}goh{}gie{}ga{}gtv{}gbn{}gkl{}/'.format(
            task_id,
            args.goh_ratio,
            args.gie_ratio,
            args.ga_ratio,
            args.gtv_ratio,
            args.gbn_ratio,
            args.gkl_ratio,
        )
        os.makedirs(tensorboard_G_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_G_path)
        ### set model: generator and fix dicriminator (classification model)
        teacher_model.eval()
        num_old_classes = args.nb_cl_fg + (task_id - 1) * args.nb_cl
        generator = get_generator(args).cuda()
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)
        CE_loss = nn.CrossEntropyLoss()
        style_loss = StyleLoss()
        loss_r_feature_layers = []
        for module in teacher_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
        ### start train for epochs
        for epoch in tqdm(range(args.epochs_G)):
            generator.train()
            for i in range(120):
                z = get_generator_initial_noise(args)
                optimizer_G.zero_grad()
                gen_imgs = generator(z)
                if args.generator_type == 'generator256':
                    gen_imgs = F.interpolate(gen_imgs, size=(224,224),mode='bilinear', align_corners=False)
                outputs_T, features_T = teacher_model(gen_imgs, out_feature=True)
                pred = outputs_T[:, :num_old_classes].data.max(1)[1]
                ### base losses
                loss_activation = -features_T.abs().mean()
                loss_one_hot = CE_loss(outputs_T[:, :num_old_classes], pred)
                softmax_o_T = torch.nn.functional.softmax(outputs_T[:, :num_old_classes], dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
                # apply total variation regularization
                diff1 = gen_imgs[:, :, :, :-1] - gen_imgs[:, :, :, 1:]
                diff2 = gen_imgs[:, :, :-1, :] - gen_imgs[:, :, 1:, :]
                diff3 = gen_imgs[:, :, 1:, :-1] - gen_imgs[:, :, :-1, 1:]
                diff4 = gen_imgs[:, :, :-1, :-1] - gen_imgs[:, :, 1:, 1:]
                # loss_tv = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
                loss_tv = (diff1.mean().abs() + diff2.mean().abs() + diff3.mean().abs() + diff4.mean().abs()) / 4
                if args.g_loss_type == 'base':
                    loss = loss_one_hot * args.goh_ratio + loss_information_entropy * args.gie_ratio \
                           + loss_activation * args.ga_ratio + loss_tv * args.gtv_ratio
                elif args.g_loss_type == 'bn':
                    loss_bn = sum([mod.r_feature for mod in loss_r_feature_layers]) / len(loss_r_feature_layers)
                    loss = loss_one_hot * args.goh_ratio + loss_information_entropy * args.gie_ratio \
                           + loss_activation * args.ga_ratio \
                           + loss_tv * args.gtv_ratio \
                           + loss_bn * args.gbn_ratio
                elif args.g_loss_type == 'bn_kl_image':
                    loss_bn = sum([mod.r_feature for mod in loss_r_feature_layers]) / len(loss_r_feature_layers)
                    ### max kl divergence
                    sample_num = args.kl_img_sample_num
                    sample1_indices = random.sample(range(0, args.batch_size_G // 2), sample_num)
                    sample2_indices = random.sample(range(args.batch_size_G // 2, args.batch_size_G), sample_num)
                    ### add kl on images
                    B, C, H, W = gen_imgs.shape
                    gen_imgs_align = gen_imgs.view([B, C * H * W])
                    sample1_outputs_T =gen_imgs_align[sample1_indices]
                    sample2_outputs_T =gen_imgs_align[sample2_indices]
                    ### add kl on feature
                    # sample1_outputs_T = outputs_T[sample1_indices]
                    # sample2_outputs_T = outputs_T[sample2_indices]
                    logp1 = F.log_softmax(sample1_outputs_T, dim=1)
                    q1    = F.softmax(sample2_outputs_T, dim=1)
                    logp2 = F.log_softmax(sample2_outputs_T, dim=1)
                    q2    = F.softmax(sample1_outputs_T, dim=1)
                    loss_kl = ((-F.kl_div(logp1, q1, size_average=False) / sample_num) + (-F.kl_div(logp2, q2, size_average=False) / sample_num)) / 2
                    loss = loss_one_hot * args.goh_ratio + loss_information_entropy * args.gie_ratio \
                           + loss_activation * args.ga_ratio \
                           + loss_tv * args.gtv_ratio \
                           + loss_bn * args.gbn_ratio \
                           + loss_kl * args.gkl_ratio
                loss.backward()
                optimizer_G.step()
            if args.g_loss_type == 'base':
                print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] " % (
                    epoch, args.epochs_G, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item()))
                writer.add_scalar('train_G/loss_oh', loss_one_hot.item(), epoch)
                writer.add_scalar('train_G/loss_ie', loss_information_entropy.item(), epoch)
                writer.add_scalar('train_G/loss_a', loss_activation.item(), epoch)
                writer.add_scalar('train_G/loss_tv', loss_tv.item(), epoch)
                if (epoch + 1) % 100 == 0:
                    torch.save(generator, args.tasks_model_path + 'epoch{}_task{}_generator'.format(epoch, task_id))
                if (epoch + 1) % 50 == 0:
                    vutils.save_image(gen_imgs[:100].clone(),tensorboard_G_path + 'output_{}.png'.format(epoch),normalize=True, nrow=10)
                    x = vutils.make_grid(gen_imgs[:100].clone(), normalize=True, scale_each=True)
                    writer.add_image('G_Image', x, epoch)
            elif args.g_loss_type == 'bn':
                print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_tv: %f] [loss_bn: %f]" % (
                    epoch, args.epochs_G, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
                    loss_tv.item(), loss_bn.item()))
                writer.add_scalar('train_G/loss_oh', loss_one_hot.item(), epoch)
                writer.add_scalar('train_G/loss_ie', loss_information_entropy.item(), epoch)
                writer.add_scalar('train_G/loss_a', loss_activation.item(), epoch)
                writer.add_scalar('train_G/loss_tv', loss_tv.item(), epoch)
                writer.add_scalar('train_G/loss_distr', loss_bn.item(), epoch)
                if (epoch + 1) % 100 == 0:
                    torch.save(generator, args.tasks_model_path + 'epoch{}_task{}_generator'.format(epoch, task_id))
                if (epoch + 1) % 10 == 0:
                    vutils.save_image(gen_imgs[:100].clone(),tensorboard_G_path + 'output_{}.png'.format(epoch + 1),normalize=True, nrow=10)
                    # x = vutils.make_grid(gen_imgs[:100].clone(), normalize=True, scale_each=True)
                    # writer.add_image('G_Image', x, epoch)
            elif args.g_loss_type == 'bn_kl_image':
                print("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_tv: %f] [loss_bn: %f] [loss_kl: %f]" % (
                    epoch, args.epochs_G, loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(),
                    loss_tv.item(), loss_bn.item(), loss_kl.item()))
                writer.add_scalar('train_G/loss_oh', loss_one_hot.item(), epoch)
                writer.add_scalar('train_G/loss_ie', loss_information_entropy.item(), epoch)
                writer.add_scalar('train_G/loss_a', loss_activation.item(), epoch)
                writer.add_scalar('train_G/loss_tv', loss_tv.item(), epoch)
                writer.add_scalar('train_G/loss_distr', loss_bn.item(), epoch)
                writer.add_scalar('train_G/loss_kl', loss_kl.item(), epoch)
                if (epoch + 1) % 100 == 0:
                    torch.save(generator, args.tasks_model_path + 'epoch{}_task{}_generator'.format(epoch, task_id))
                if (epoch + 1) % 10 == 0:
                    vutils.save_image(gen_imgs[:100].clone(),tensorboard_G_path + 'output_{}.png'.format(epoch + 1),normalize=True, nrow=10)
    return generator