#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
import copy
import argparse
try:
    import cPickle as pickle
except:
    import pickle
from utils.compute_accuracy import test_ac
from utils.general import *
from trainers.df_generator_trainer import train_df_generator
from models import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datasets.get_dataset import get_select_dataset, get_cur_dataloader

### args settings for classification ###
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100','tinyimagenet','cub_imagenet','cub_200_2011'])
parser.add_argument('--dataset_dir', default='./data/cifar-100-python', type=str)
parser.add_argument('--model', type=str, default='resnet34', choices=['resnet34','resnet18_torchvision_model','resnet18_modify'])
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--nb_cl_fg', default=20, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=20, type=int, help='Classes per group')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--T', default=2, type=float, help='Temperature for distialltion')
parser.add_argument('--random_seed', default=1988, type=int, help='random seed')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--val_epoch', default=1, type=int, help='Epochs')
parser.add_argument('--train_batch_size', default=128, type=int, help='Batch size for train')
parser.add_argument('--test_batch_size', default=100, type=int, help='Batch size for test')
parser.add_argument('--eval_batch_size', default=100, type=int, help='Batch size for eval')
parser.add_argument('--custom_weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--custom_momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--save_epoch', default=50, type=int, help='save the model in every save_epoch')
parser.add_argument('--lr_strat', default=[5, 10], nargs="+", type=int, help='Epochs where learning rate gets decreased')
parser.add_argument('--lr_factor', default=0.1, type=float, help='Learning rate decrease factor')
parser.add_argument('--epochs', default=1, type=int, help='Epochs')
parser.add_argument('--base_lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--tensorboard', default=False, action="store_true")
parser.add_argument('--plot_cm', default=False, action="store_true")
parser.add_argument('--load_T', default=False, action="store_true")
parser.add_argument('--load_T_only_imagenet_pretrain', default=False, action="store_true")
parser.add_argument('--load_resume_T', default=False, action="store_true")
parser.add_argument('--load_T_name', default='ResNet34_Model_run_0_step_0_84.2.pth', type=str)
parser.add_argument('--load_resume_T_name', default='ResNet34_Model_run_1_step_0_84.2.pth', type=str)
parser.add_argument('--c_resume_iteration', type=int, default=0)
parser.add_argument('--scheduler', type=str, default='cosWR', choices=['1cyc', 'cosWR', 'None', 'MultiStep'])
parser.add_argument('--method', type=str, default='dfkd', choices=['finetune','dfkd','lwf', 'oracle'])
parser.add_argument('--c_loss_type', type=str, default='ce', choices=['ndkd', 'gdkd','gnkd_ncecut','gnce','ce'])
parser.add_argument('--dataloader_type', type=str, default='il', choices=['il', 'oracle'])
parser.add_argument('--loss_ratio_adaptive', default=False, action="store_true")
parser.add_argument('--o_ce', default=0, type=float, help='loss ratio for old data CE loss')
parser.add_argument('--o_kd', default=0, type=float, help='loss ratio for old data kd loss')
parser.add_argument('--n_ce', default=0, type=float, help='loss ratio for new data CE loss')
parser.add_argument('--n_kd', default=0, type=float, help='loss ratio for new data kd loss')
parser.add_argument('--toy_example', default=False, action="store_true")
### args setting for DFKD generator ###
parser.add_argument('--generator_type', type=str, default='ce', choices=['generator32', 'generator128','generator256','unet'])
parser.add_argument('--load_G', default=False, action="store_true")
parser.add_argument('--load_G_name', default='epoch500_task1_generator_gkl0.1', type=str)
parser.add_argument('--epochs_G', type=int, default=1000, metavar='N', help='number of epochs to train')
parser.add_argument('--batch_size_G', type=int, default=512)
parser.add_argument('--g_resume_iteration', type=int, default=0)
parser.add_argument('--tn_batch_size_G', type=int, default=128, help='train new task batch size G')
parser.add_argument('--gtnbs_adaptive', default=False, action="store_true")
parser.add_argument('--latent_dim_G', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--img_size_G', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels_G', type=int, default=3, help='number of image channels')
parser.add_argument('--lr_G', type=float, default=0.01, metavar='LR', help='generator learning rate')
parser.add_argument('--ga_ratio', type=float, default=0.1)
parser.add_argument('--goh_ratio', type=float, default=1)
parser.add_argument('--gie_ratio', type=float, default=5)
### for different generator loss
parser.add_argument('--g_loss_type', type=str, default='bn', choices=['base', 'bn', 'bn_kl_image','gnkd_ncecut','gnce','ce'])
parser.add_argument('--kl_img_sample_num', type=int, default=200, help='the numner of kl image sample image for kl loss')
parser.add_argument('--gtv_ratio', type=float, default=0)
parser.add_argument('--gbn_ratio', type=float, default=0, help='deepinversion batchnorm loss ratio')
parser.add_argument('--gkl_ratio', type=float, default=0)
parser.add_argument('--toy_G', default=False, action="store_true")
args = parser.parse_args()
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)

###  get whole dataset ###
trainset, testset, evalset, X_train_total, Y_train_total, X_valid_total, Y_valid_total = get_select_dataset(args)
### Launch the different runs ###
for n_run in range(args.nb_runs):
    ### set random seed, log, IL order and IL steps
    set_random_seed(args,n_run)
    args.iteration_num = set_iteration_number(args)
    set_and_write_log(args)
    print("Generating orders")
    order = np.arange(args.num_classes)
    order_list = list(order)
    print(__file__); print('Settings:'); print(order_list); print(vars(args))
    ### start incremental step train
    for iteration in range(0, int(args.iteration_num)):
        if iteration == 0:
            ### get current il step dataloader
            trainloader, testloader = get_cur_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total, Y_valid_total)
            ### set model
            if args.model == 'resnet34':
                tg_model = ResNet34(num_classes=args.nb_cl_fg).cuda()
            elif args.model == 'resnet18_torchvision_model':
                tg_model = torchvision.models.resnet18(pretrained=True).cuda()
                in_features = tg_model.fc.in_features
                tg_model.fc = torch.nn.Linear(in_features, args.nb_cl_fg).cuda()
            elif args.model == 'resnet18_modify':
                tg_model = resnet18(pretrained=True).cuda()
                in_features = tg_model.linear.in_features
                tg_model.linear = torch.nn.Linear(in_features, args.nb_cl_fg).cuda()
            if args.load_T:
                if args.load_T_only_imagenet_pretrain:
                    pass
                else:
                    pretrain_dir = os.path.join(args.teacher_model_path, args.load_T_name)
                    tg_model.load_state_dict(torch.load(pretrain_dir))
                    test_ac(tg_model, X_valid_total, Y_valid_total,evalset, testloader, order, iteration, args)
            else:
                print("Train the model for iteration {}".format(iteration))
                tg_optimizer = optim.SGD(tg_model.parameters(), lr=args.base_lr, momentum=args.custom_momentum, weight_decay=args.custom_weight_decay)
                if args.scheduler == 'cosWR':
                    tg_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(tg_optimizer, args.epochs // 1, 1)
                elif args.scheduler == '1cyc':
                    tg_lr_scheduler = optim.lr_scheduler.OneCycleLR(tg_optimizer, max_lr=args.base_lr, steps_per_epoch=len(trainloader), epochs=args.epochs)
                elif args.scheduler == 'MultiStep':
                    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat, gamma=args.lr_factor)
                cls_criterion = nn.CrossEntropyLoss().cuda()
                for epoch in tqdm(range(args.epochs)):
                    tg_model.train()
                    print('LR:', tg_lr_scheduler.get_last_lr()[0])
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                        tg_optimizer.zero_grad()
                        outputs = tg_model(inputs)
                        loss_cls = cls_criterion(outputs[:, 0:args.nb_cl_fg], targets)
                        loss = loss_cls
                        loss.backward()
                        tg_optimizer.step()
                        if args.scheduler == 'cosWR':
                            tg_lr_scheduler.step(epoch + batch_idx / len(trainloader))
                        elif args.scheduler == '1cyc':
                            tg_lr_scheduler.step()
                    if args.scheduler == 'MultiStep':
                        tg_lr_scheduler.step()
                    print('Epoch: %d, loss_cls: %.4f' % (epoch, loss_cls.item()))
                    if (epoch + 1) % args.val_epoch == 0:
                        test_ac(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args)
                    if (epoch + 1) % args.save_epoch == 0:
                        ckp_name = os.path.join(args.tasks_model_path + '{}_run_{}_step_{}.pth').format(args.model, n_run, iteration)
                        torch.save(tg_model.state_dict(), ckp_name)
        else:
            print("Train the model for iteration {}".format(iteration))
            ### get current  dataloader: il or oracle
            trainloader, testloader = get_cur_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total, Y_valid_total)
            ### train new model ###
            ### set reference model
            ref_model = copy.deepcopy(tg_model).cuda()
            for param in ref_model.parameters():
                param.requires_grad = False
            ### set new classifier model
            tg_model, num_old_classes = set_new_classifier(ref_model, tg_model, iteration, args)
            ### load resume or train new model ###
            if args.load_resume_T and iteration == args.c_resume_iteration:
                pretrain_dir = os.path.join(args.teacher_model_path, args.load_resume_T_name)
                tg_model.load_state_dict(torch.load(pretrain_dir))
                test_ac(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args)
            elif iteration > args.c_resume_iteration:
                if args.method == 'finetune':
                    ### set tensorboard path
                    tb_path = args.tensorboard_base_path + 'task{}/'.format(iteration)
                    os.makedirs(tb_path, exist_ok=True)
                    writer = SummaryWriter(tb_path)
                    ### get optimizer, scheduler
                    tg_optimizer = optim.SGD(tg_model.parameters(), lr=args.base_lr, momentum=args.custom_momentum,weight_decay=args.custom_weight_decay)
                    if args.scheduler == 'cosWR':
                        tg_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(tg_optimizer, args.epochs // 1, 1)
                    elif args.scheduler == '1cyc':
                        tg_lr_scheduler = optim.lr_scheduler.OneCycleLR(tg_optimizer, max_lr=args.base_lr,steps_per_epoch=len(trainloader),epochs=args.epochs)
                    elif args.scheduler == 'MultiStep':
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat,gamma=args.lr_factor)
                    cls_criterion = nn.CrossEntropyLoss().cuda()
                    ### start train for epochs
                    for epoch in tqdm(range(args.epochs)):
                        if args.tensorboard:
                            writer.add_scalar('lr', tg_lr_scheduler.get_last_lr()[0], epoch)
                        tg_model.train()
                        print('LR:', tg_lr_scheduler.get_last_lr()[0])
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            tg_optimizer.zero_grad()
                            outputs = tg_model(inputs)
                            loss = cls_criterion(outputs, targets)
                            loss.backward()
                            tg_optimizer.step()
                            if args.scheduler == 'cosWR':
                                tg_lr_scheduler.step(epoch + batch_idx / len(trainloader))
                            elif args.scheduler == '1cyc':
                                tg_lr_scheduler.step()
                        if args.scheduler == 'MultiStep':
                            tg_lr_scheduler.step()
                        print('Epoch: %d, loss: %.4f' % (epoch, loss.item()))
                        if (epoch + 1) % args.val_epoch == 0:
                            test_ac(tg_model, X_valid_total, Y_valid_total,evalset, testloader, order, iteration, args)
                        if (epoch + 1) % args.save_epoch == 0:
                            ckp_name = os.path.join(args.tasks_model_path + 'ResNet34_Model_run_{}_step_{}.pth').format(n_run, iteration)
                            torch.save(tg_model.state_dict(), ckp_name)
                elif args.method == 'lwf':
                    ### set tensorboard path
                    tb_path = args.tensorboard_base_path + 'task{}/'.format(iteration)
                    os.makedirs(tb_path, exist_ok=True)
                    writer = SummaryWriter(tb_path)
                    ### get optimizer, scheduler
                    tg_optimizer = optim.SGD(tg_model.parameters(), lr=args.base_lr, momentum=args.custom_momentum,weight_decay=args.custom_weight_decay)
                    if args.scheduler == 'cosWR':
                        tg_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(tg_optimizer, args.epochs // 1, 1)
                    elif args.scheduler == '1cyc':
                        tg_lr_scheduler = optim.lr_scheduler.OneCycleLR(tg_optimizer, max_lr=args.base_lr,steps_per_epoch=len(trainloader),epochs=args.epochs)
                    elif args.scheduler == 'MultiStep':
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat,gamma=args.lr_factor)
                    cls_criterion = nn.CrossEntropyLoss().cuda()
                    ### start train for epochs
                    for epoch in tqdm(range(args.epochs)):
                        if args.tensorboard:
                            writer.add_scalar('lr', tg_lr_scheduler.get_last_lr()[0], epoch)
                        tg_model.train()
                        print('LR:', tg_lr_scheduler.get_last_lr()[0])
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            tg_optimizer.zero_grad()
                            old_outputs = ref_model(inputs)
                            outputs = tg_model(inputs)
                            targets = set_targets_cut(args, targets, iteration)
                            loss_cls_new = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)],targets)
                            # distillation loss for main classifier
                            soft_target = F.softmax(old_outputs[:, :num_old_classes] / args.T, dim=1)
                            logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                            loss_distill_new = -torch.mean(torch.sum(soft_target * logp, dim=1))
                            if args.loss_ratio_adaptive:
                                if args.nb_cl_fg == args.nb_cl:
                                    alpha = float(iteration) / float(iteration + 1)
                                else:
                                    alpha = float(args.nb_cl_fg / args.nb_cl + iteration - 1) / float(args.nb_cl_fg / args.nb_cl + iteration)
                                loss = (1.0 - alpha) * loss_cls_new + alpha * loss_distill_new
                            else:
                                loss = args.n_ce * loss_cls_new + args.n_kd * loss_distill_new

                            loss.backward()
                            tg_optimizer.step()
                            if args.scheduler == 'cosWR':
                                tg_lr_scheduler.step(epoch + batch_idx / len(trainloader))
                            elif args.scheduler == '1cyc':
                                tg_lr_scheduler.step()
                        if args.scheduler == 'MultiStep':
                            tg_lr_scheduler.step()
                        print('Epoch: %d, loss: %.4f loss cls new: %.4f loss distill new: %.4f' % (epoch, loss.item(), loss_cls_new.item(), loss_distill_new.item()))
                        if (epoch + 1) % args.val_epoch == 0:
                            test_ac(tg_model, X_valid_total, Y_valid_total,evalset, testloader, order, iteration, args)
                        if (epoch + 1) % args.save_epoch == 0:
                            ckp_name = os.path.join(args.tasks_model_path + '{}_run_{}_step_{}.pth').format(args.model, n_run, iteration)
                            torch.save(tg_model.state_dict(), ckp_name)
                elif args.method == 'oracle':
                    ### set tensorboard path
                    tb_path = args.tensorboard_base_path + 'task{}/'.format(iteration)
                    os.makedirs(tb_path, exist_ok=True)
                    writer = SummaryWriter(tb_path)
                    ### get optimizer, scheduler
                    tg_optimizer = optim.SGD(tg_model.parameters(), lr=args.base_lr, momentum=args.custom_momentum,weight_decay=args.custom_weight_decay)
                    if args.scheduler == 'cosWR':
                        tg_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(tg_optimizer, args.epochs // 1, 1)
                    elif args.scheduler == '1cyc':
                        tg_lr_scheduler = optim.lr_scheduler.OneCycleLR(tg_optimizer, max_lr=args.base_lr,steps_per_epoch=len(trainloader),epochs=args.epochs)
                    elif args.scheduler == 'MultiStep':
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat,gamma=args.lr_factor)
                    cls_criterion = nn.CrossEntropyLoss().cuda()
                    ### start train for epochs
                    for epoch in tqdm(range(args.epochs)):
                        if args.tensorboard:
                            writer.add_scalar('lr', tg_lr_scheduler.get_last_lr()[0], epoch)
                        tg_model.train()
                        print('LR:', tg_lr_scheduler.get_last_lr()[0])
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            tg_optimizer.zero_grad()
                            old_outputs = ref_model(inputs)
                            outputs = tg_model(inputs)
                            loss = cls_criterion(outputs[:, 0:(args.nb_cl_fg + iteration * args.nb_cl)], targets)
                            loss.backward()
                            tg_optimizer.step()
                            if args.scheduler == 'cosWR':
                                tg_lr_scheduler.step(epoch + batch_idx / len(trainloader))
                            elif args.scheduler == '1cyc':
                                tg_lr_scheduler.step()
                        if args.scheduler == 'MultiStep':
                            tg_lr_scheduler.step()
                        print('Epoch: %d, loss_cls: %.4f' % (epoch, loss.item()))
                        if (epoch + 1) % args.val_epoch == 0:
                            test_ac(tg_model, X_valid_total, Y_valid_total,evalset, testloader, order, iteration, args)
                        if (epoch + 1) % args.save_epoch == 0:
                            ckp_name = os.path.join(args.tasks_model_path + '{}_run_{}_step_{}.pth').format(args.model, n_run, iteration)
                            torch.save(tg_model.state_dict(), ckp_name)
                elif args.method == 'dfkd':
                    ### set tensorboard path
                    tb_path = args.tensorboard_base_path + 'task{}/'.format(iteration)
                    os.makedirs(tb_path, exist_ok=True)
                    writer = SummaryWriter(tb_path)
                    ### train generator first ###
                    generator = train_df_generator(ref_model, iteration, args)
                    ### set bathsize of G images ###
                    if args.gtnbs_adaptive:
                        args.tn_batch_size_G = round((iteration * 5000)/len(trainloader))
                    ### train new model ###
                    ### get optimizer, scheduler
                    tg_optimizer = optim.SGD(tg_model.parameters(), lr=args.base_lr, momentum=args.custom_momentum,weight_decay=args.custom_weight_decay)
                    if args.scheduler == 'cosWR':
                        tg_lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(tg_optimizer, args.epochs // 1, 1)
                    elif args.scheduler == '1cyc':
                        tg_lr_scheduler = optim.lr_scheduler.OneCycleLR(tg_optimizer, max_lr=args.base_lr,steps_per_epoch=len(trainloader),epochs=args.epochs)
                    elif args.scheduler == 'MultiStep':
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=args.lr_strat,gamma=args.lr_factor)
                    cls_criterion = nn.CrossEntropyLoss().cuda()
                    ### start train for epochs
                    for epoch in tqdm(range(args.epochs)):
                        if args.tensorboard:
                            writer.add_scalar('lr', tg_lr_scheduler.get_last_lr()[0], epoch)
                        tg_model.train()
                        print('LR:', tg_lr_scheduler.get_last_lr()[0])
                        for batch_idx, (inputs, targets) in enumerate(trainloader):
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            tg_optimizer.zero_grad()
                            if args.c_loss_type == 'ndkd':
                                old_outputs = ref_model(inputs)
                                outputs = tg_model(inputs)
                                ### distillation loss for main classifier
                                soft_target = F.softmax(old_outputs[:, :num_old_classes] / args.T, dim=1)
                                logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_new = -torch.mean(torch.sum(soft_target * logp, dim=1))
                                loss = loss_distill_new
                            elif args.c_loss_type == 'gdkd':
                                with torch.no_grad():
                                    z = get_generator_initial_noise(args)
                                    gen_imgs = generator(z)
                                    old_gen_outputs = ref_model(gen_imgs)
                                gen_outputs = tg_model(gen_imgs.detach())
                                ### distillation loss for generative old data
                                gen_soft_target = F.softmax(old_gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                gen_logp = F.log_softmax(gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_gen = -torch.mean(torch.sum(gen_soft_target * gen_logp, dim=1))
                                loss = loss_distill_gen
                            elif args.c_loss_type == 'gnkd_ncecut':
                                ### forward for generative old data
                                with torch.no_grad():
                                    z = get_generator_initial_noise(args)
                                    gen_imgs = generator(z)
                                    if args.generator_type == 'generator256':
                                        gen_imgs = F.interpolate(gen_imgs, size=(224, 224), mode='bilinear',align_corners=False)
                                    old_gen_outputs = ref_model(gen_imgs)
                                gen_outputs = tg_model(gen_imgs.detach())
                                outputs = tg_model(inputs)
                                ### distillation loss for generative old data
                                gen_soft_target = F.softmax(old_gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                gen_logp = F.log_softmax(gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_gen = -torch.mean(torch.sum(gen_soft_target * gen_logp, dim=1))
                                ### loss for new data
                                old_outputs = ref_model(inputs)
                                targets = set_targets_cut(args, targets, iteration)
                                loss_cls_new = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)],targets)
                                ### distillation loss for new data
                                soft_target = F.softmax(old_outputs[:, :num_old_classes] / args.T, dim=1)
                                logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_new = -torch.mean(torch.sum(soft_target * logp, dim=1))
                                if args.loss_ratio_adaptive:
                                    if args.nb_cl_fg == args.nb_cl:
                                        alpha = float(iteration) / float(iteration + 1)
                                    else:
                                        alpha = float(args.nb_cl_fg / args.nb_cl + iteration - 1) / float(
                                            args.nb_cl_fg / args.nb_cl + iteration)
                                    loss = (1.0 - alpha) * loss_cls_new + alpha * (loss_distill_gen + loss_distill_new)
                                else:
                                    loss = args.n_ce * loss_cls_new + args.o_kd * loss_distill_gen + args.n_kd * loss_distill_new
                            elif args.c_loss_type == 'gnkdcecut_ncecut':
                                ### forward for generative old data
                                with torch.no_grad():
                                    z = get_generator_initial_noise(args)
                                    gen_imgs = generator(z)
                                    old_gen_outputs = ref_model(gen_imgs)
                                gen_outputs = tg_model(gen_imgs.detach())
                                outputs = tg_model(inputs)
                                ### distillation loss for generative old data
                                gen_soft_target = F.softmax(old_gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                gen_logp = F.log_softmax(gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_gen = -torch.mean(torch.sum(gen_soft_target * gen_logp, dim=1))
                                ### ce loss for generative old data
                                old_gen_outputs_sm = F.softmax(old_gen_outputs, dim=1)
                                gen_values, gen_targets = old_gen_outputs_sm.max(1)
                                gen_outputs = tg_model(gen_imgs.detach())
                                loss_cls_old = cls_criterion(gen_outputs[:, 0:num_old_classes], gen_targets)
                                ### ce loss for new data
                                old_outputs = ref_model(inputs)
                                targets = set_targets_cut(args, targets, iteration)
                                loss_cls_new = cls_criterion(outputs[:, num_old_classes:(num_old_classes + args.nb_cl)],targets)
                                ### distillation loss for new data
                                soft_target = F.softmax(old_outputs[:, :num_old_classes] / args.T, dim=1)
                                logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_new = -torch.mean(torch.sum(soft_target * logp, dim=1))
                                if args.loss_ratio_adaptive:
                                    alpha = float(iteration) / float(iteration + 1)
                                    loss = (1.0 - alpha) * (loss_cls_old + loss_cls_new) + alpha * (loss_distill_gen + loss_distill_new)
                                else:
                                    loss = args.o_ce * loss_cls_old + args.n_ce * loss_cls_new + args.o_kd * loss_distill_gen + args.n_kd * loss_distill_new
                            elif args.c_loss_type == 'gnkdce_nce':  # old_new_no_cut_ce
                                ### forward for generative old data
                                with torch.no_grad():
                                    z = get_generator_initial_noise(args)
                                    gen_imgs = generator(z)
                                    old_gen_outputs = ref_model(gen_imgs)
                                gen_outputs = tg_model(gen_imgs.detach())
                                outputs = tg_model(inputs)
                                ### distillation loss for generative old data
                                gen_soft_target = F.softmax(old_gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                gen_logp = F.log_softmax(gen_outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_gen = -torch.mean(torch.sum(gen_soft_target * gen_logp, dim=1))
                                ### ce loss for generative old data
                                old_gen_outputs_sm = F.softmax(old_gen_outputs, dim=1)
                                gen_values, gen_targets = old_gen_outputs_sm.max(1)
                                gen_outputs = tg_model(gen_imgs.detach())
                                loss_cls_old = cls_criterion(gen_outputs[:, 0:num_old_classes + args.nb_cl], gen_targets)
                                ### ce loss for new data
                                old_outputs = ref_model(inputs)
                                loss_cls_new = cls_criterion(outputs[:, 0:(num_old_classes + args.nb_cl)], targets)
                                ### distillation loss for new data
                                soft_target = F.softmax(old_outputs[:, :num_old_classes] / args.T, dim=1)
                                logp = F.log_softmax(outputs[:, :num_old_classes] / args.T, dim=1)
                                loss_distill_new = -torch.mean(torch.sum(soft_target * logp, dim=1))
                                if args.loss_ratio_adaptive:
                                    alpha = float(iteration) / float(iteration + 1)
                                    loss = (1.0 - alpha) * (loss_cls_old + loss_cls_new) + alpha * (loss_distill_gen + loss_distill_new)
                                else:
                                    loss = args.o_ce * loss_cls_old + args.n_ce * loss_cls_new + args.o_kd * loss_distill_gen + args.n_kd * loss_distill_new
                            elif args.c_loss_type == 'gnce':
                                ### forward for generative old data
                                with torch.no_grad():
                                    z = Variable(torch.randn(args.tn_batch_size_G, args.latent_dim_G)).cuda()
                                    gen_imgs = generator(z)
                                    old_gen_outputs = ref_model(gen_imgs)
                                ### ce loss for generative old data
                                old_gen_outputs_sm = F.softmax(old_gen_outputs, dim=1)
                                gen_values, gen_targets = old_gen_outputs_sm.max(1)
                                gen_outputs = tg_model(gen_imgs.detach())
                                loss_cls_old = cls_criterion(gen_outputs[:, 0:(num_old_classes + args.nb_cl)], gen_targets)

                                ### forward for new task data
                                outputs = tg_model(inputs)
                                ### ce loss for new data
                                loss_cls_new = cls_criterion(outputs[:, 0:(num_old_classes + args.nb_cl)], targets)
                                ### loss_cls_new = torch.zeros(1).cuda()
                                if args.loss_ratio_adaptive:
                                    alpha = float(iteration) / float(iteration + 1)
                                    loss = alpha * loss_cls_old + (1.0 - alpha) * loss_cls_new
                                else:
                                    loss = args.o_ce * loss_cls_old + args.n_ce * loss_cls_new
                            loss.backward()
                            tg_optimizer.step()
                            if args.scheduler == 'cosWR':
                                tg_lr_scheduler.step(epoch + batch_idx / len(trainloader))
                            elif args.scheduler == '1cyc':
                                tg_lr_scheduler.step()
                        if args.scheduler == 'MultiStep':
                            tg_lr_scheduler.step()

                        ### print different loss for each epoch
                        if args.c_loss_type == 'ndkd':
                            print('Epoch: %d, loss_distill_new: %.4f' % (epoch, loss_distill_new.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/nkd', loss_distill_new.item(), epoch)
                        elif args.c_loss_type == 'gdkd':
                            print('Epoch: %d, loss_distill_gen: %.4f' % (epoch, loss_distill_gen.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/gkd', loss_distill_gen.item(), epoch)
                        elif args.c_loss_type == 'gnkd_ncecut':
                            print('Epoch: %d, loss_cls_new: %.4f, loss_distill_gen: %.4f, loss_distill_new: %.4f' % (
                                epoch, loss_cls_new.item(), loss_distill_gen.item(), loss_distill_new.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/gkd', loss_distill_gen.item(), epoch)
                                writer.add_scalar('loss/nce', loss_cls_new.item(), epoch)
                                writer.add_scalar('loss/nkd', loss_distill_new.item(), epoch)
                        elif args.c_loss_type == 'gnkdcecut_ncecut':
                            print('Epoch: %d, loss_cls_old: %.4f, loss_cls_new: %.4f, loss_distill_gen: %.4f, loss_distill_new: %.4f' % (
                                    epoch, loss_cls_old.item(), loss_cls_new.item(), loss_distill_gen.item(),
                                    loss_distill_new.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/gce', loss_cls_old.item(), epoch)
                                writer.add_scalar('loss/gkd', loss_distill_gen.item(), epoch)
                                writer.add_scalar('loss/nce', loss_cls_new.item(), epoch)
                                writer.add_scalar('loss/nkd', loss_distill_new.item(), epoch)
                        elif args.c_loss_type == 'gnkdce_nce':
                            print('Epoch: %d, loss_cls_old: %.4f, loss_cls_new: %.4f, loss_distill_gen: %.4f, loss_distill_new: %.4f' % (
                                    epoch, loss_cls_old.item(), loss_cls_new.item(), loss_distill_gen.item(),
                                    loss_distill_new.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/gce', loss_cls_old.item(), epoch)
                                writer.add_scalar('loss/gkd', loss_distill_gen.item(), epoch)
                                writer.add_scalar('loss/nce', loss_cls_new.item(), epoch)
                                writer.add_scalar('loss/nkd', loss_distill_new.item(), epoch)
                        elif args.c_loss_type == 'gnce':
                            print('Epoch: %d, loss_cls_old: %.4f, loss_cls_new: %.4f' % (
                                epoch, loss_cls_old.item(), loss_cls_new.item()))
                            if args.tensorboard:
                                writer.add_scalar('loss/oce', loss_cls_old.item(), epoch)
                                writer.add_scalar('loss/nce', loss_cls_new.item(), epoch)

                        ### evaluate the val set and save model
                        acc_old, acc_new, acc_total = test_ac(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args)
                        writer.add_scalar('acc/acc_old', acc_old, epoch)
                        writer.add_scalar('acc/acc_new', acc_new, epoch)
                        writer.add_scalar('acc/acc_total', acc_total, epoch)
                        if (epoch + 1) % args.save_epoch == 0:
                            ckp_name = os.path.join(args.tasks_model_path + '{}_run_{}_step_{}.pth').format(args.model, n_run, iteration)
                            torch.save(tg_model.state_dict(), ckp_name)



























### for save image
        # gen_preds = old_gen_outputs[:, :num_old_classes].data.max(1)[1]
        # for id in range(gen_imgs.shape[0]):
        #     if not os.path.exists(
        #             tb_path + 'G_images{}/class{:03d}'.format(num_old_classes, gen_preds[id])):
        #         os.makedirs(
        #             tb_path + 'G_images{}/class{:03d}'.format(num_old_classes, gen_preds[id]))
        #     place_to_store = tb_path + 'G_images{}/class{:03d}/img_indexid{:05d}.jpg'.format(
        #         num_old_classes, gen_preds[id], id)
        #     from torchvision.utils import save_image
        #
        #     save_image(gen_imgs[id], place_to_store, normalize=True, scale_each=True)





