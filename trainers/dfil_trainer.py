import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from DFIL.util.meters import AverageMeter
from DFIL.util.general import *
from DFIL.util.general import get_device
from collections import OrderedDict, defaultdict
import torch.optim as optim

class DFILTrainer(BaseTrainer):
    def __init__(self, teacher, generator, student, new_tasks_model, CE_loss, LSR_loss, KD_loss, noise_dim, num_classes,
                 ratios, update_freqs, batch_size, batches_per_epoch,
                 config, test_loader, current_classes, old_classes,
                 logdir=None):
        super(BaseTrainer, self).__init__()
        self.teacher = teacher
        self.generator = generator
        self.student = student
        self.new_tasks_model = new_tasks_model
        self.CE_loss = CE_loss
        self.LSR_loss = LSR_loss
        self.KD_loss = KD_loss
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.a_ratio, self.oh_ratio, self.ie_ratio, self.ce_ratio, self.nkd_ratio, self.att_ratio = ratios
        self.update_freq_G, self.update_freq_S, self.update_freq_SO, self.update_freq_SN = update_freqs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.logdir = logdir

        # for test
        self.config = config
        self.test_loader = test_loader
        self.current_classes = current_classes
        self.old_classes = old_classes

    def train_raw(self, epoch, old_tasks_dataloader, new_tasks_dataloader, optimizer_G, optimizer_S, log_interval=100, schedulers=None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()

        G_CE_loss, G_A_loss, G_AKD_loss, \
        SO_CE_loss, SO_KD_loss, \
        SN_CE_loss, SN_KD_loss = \
            AverageMeter(), AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter()
        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            for old_batch_idx, (old_xs, old_ys) in enumerate(old_tasks_dataloader):
                if batch_idx == old_batch_idx:
                    self.student.train()

                    old_xs = old_xs.to(get_device(self.config.cuda))
                    old_ys = old_ys.to(get_device(self.config.cuda))
                    new_xs = new_xs.to(get_device(self.config.cuda))
                    new_ys = new_ys.to(get_device(self.config.cuda))

                    outputs_T, features_T = self.teacher(old_xs, out_feature=True)

                    optimizer_S.zero_grad()
                    old_xs_outputs_S = self.student(old_xs)
                    new_xs_outputs_S = self.student(new_xs)
                    old_xs_outputs_T = self.teacher(old_xs, out_feature=False)
                    new_xs_outputs_T = self.teacher(new_xs, out_feature=False)
                    old_xs_T_label = old_xs_outputs_T[:, :len(self.old_classes)].data.max(1)[1]

                    old_data_CE_loss = self.CE_loss(old_xs_outputs_S[:, :len(self.old_classes) + len(self.current_classes)],
                                                    old_xs_T_label)
                    old_data_KD_loss = self.KD_loss(old_xs_outputs_S[:, :len(self.old_classes)], old_xs_outputs_T[:, :len(self.old_classes)].detach())
                    new_data_KD_loss = self.KD_loss(new_xs_outputs_S[:, :len(self.old_classes)],
                                                    new_xs_outputs_T[:, :len(self.old_classes)].detach())
                    # new_data_CE_loss = F.cross_entropy(new_xs_outputs_S[:, len(self.old_classes):len(self.old_classes) + len(self.current_classes)],
                    #                                 new_ys - len(self.old_classes), reduction="mean")
                    new_data_CE_loss = F.cross_entropy(new_xs_outputs_S[:, :len(self.old_classes)+len(self.current_classes)],
                                                    new_ys, reduction="mean")
                    loss_S = self.config.odce_ratio * old_data_CE_loss + self.config.odkd_ratio * old_data_KD_loss \
                         + self.config.ndce_ratio * new_data_CE_loss + self.config.ndkd_ratio * new_data_KD_loss

                    loss_S.backward()
                    optimizer_S.step()

                    SO_CE_loss.update(old_data_CE_loss.item())
                    SO_KD_loss.update(old_data_KD_loss.item())
                    SN_CE_loss.update(new_data_CE_loss.item())
                    SN_KD_loss.update(new_data_KD_loss.item())

                        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        #         epoch, batch_idx * len(new_xs), len(new_tasks_dataloader.dataset),
                        #             100. * batch_idx / len(new_tasks_dataloader), new_data_loss.item()))

                    if schedulers is not None:
                        def step_scheduler(scheduler):
                            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                                scheduler.step(epoch - 1 + batch_idx / self.batches_per_epoch)
                            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                                scheduler.step()

                        if isinstance(schedulers, tuple):
                            for scheduler in schedulers:
                                step_scheduler(scheduler)
                        else:
                            step_scheduler(schedulers)

                    if (batch_idx + 1) % log_interval == 0:
                        # print(alpha)
                        t1 = time.time()
                        t_epoch = t1 - t0
                        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
                              'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
                              'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
                              format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                                     SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
                    # self.test_old_and_new()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
              'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
              'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
              format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                     SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
        #self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg, SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg

    def train(self, epoch, new_tasks_dataloader, optimizer_G, optimizer_S, log_interval=100, schedulers=None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()

        G_CE_loss, G_A_loss, G_AKD_loss, \
        SO_CE_loss, SO_KD_loss, \
        SN_CE_loss, SN_KD_loss = \
            AverageMeter(), AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter()
        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            self.student.train()
            z = torch.randn([self.batch_size, self.noise_dim]).to(get_device(self.config.cuda))
            #GT_label = torch.randint(0, self.num_classes, [self.batch_size]).to(get_device(self.config.cuda))
            GT_label = torch.randint(0, len(self.old_classes), [self.batch_size]).to(get_device(self.config.cuda))

            new_xs = new_xs.to(get_device(self.config.cuda))
            new_ys = new_ys.to(get_device(self.config.cuda))

            optimizer_G.zero_grad()
            # gen_imgs = self.generator(z, GT_label)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
            gen_imgs = self.generator(z)
            outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)

            # generator
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_G:
                #print("update_G_batch_id{}".format(batch_idx))
                # loss_activation = -features_T.abs().mean()
                # loss_classification = self.LSR_loss(outputs_T, GT_label)
                # outputs_S = self.student(gen_imgs)
                # loss_adv_student = -self.KD_loss(outputs_S, outputs_T)
                # T_label = outputs_T.data.max(1)[1]
                # loss_onehot = self.CE_loss(outputs_T, T_label)
                # softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
                # loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_activation = -features_T.abs().mean()
                loss_classification = self.LSR_loss(outputs_T[:, :len(self.old_classes)], GT_label)
                outputs_S = self.student(gen_imgs)
                loss_adv_student = -self.KD_loss(outputs_S[:, :len(self.old_classes)], outputs_T[:, :len(self.old_classes)])
                T_label = outputs_T[:, :len(self.old_classes)].data.max(1)[1]
                loss_onehot = self.CE_loss(outputs_T[:, :len(self.old_classes)], T_label)
                softmax_o_T = torch.nn.functional.softmax(outputs_T[:, :len(self.old_classes)], dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_G = loss_classification * self.ce_ratio + loss_onehot * self.oh_ratio + loss_information_entropy * self.ie_ratio + loss_activation * self.a_ratio + loss_adv_student * self.nkd_ratio
                loss_G.backward()
                optimizer_G.step()

                G_CE_loss.update(loss_classification.item())
                G_A_loss.update(loss_activation.item())
                G_AKD_loss.update(loss_adv_student)



            # student
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_S:
                #print("update_S_batch_id{}".format(batch_idx))
                optimizer_S.zero_grad()
                old_xs_outputs_S = self.student(gen_imgs.detach())
                new_xs_outputs_S = self.student(new_xs)
                old_xs_outputs_T = self.teacher(gen_imgs.detach(), out_feature=False)
                new_xs_outputs_T = self.teacher(new_xs, out_feature=False)
                old_xs_T_label = old_xs_outputs_T[:, :len(self.old_classes)].data.max(1)[1]

                old_data_CE_loss = self.CE_loss(old_xs_outputs_S[:, :len(self.old_classes) + len(self.current_classes)],
                                                old_xs_T_label)
                old_data_KD_loss = self.KD_loss(old_xs_outputs_S[:, :len(self.old_classes)], old_xs_outputs_T[:, :len(self.old_classes)].detach())
                new_data_KD_loss = self.KD_loss(new_xs_outputs_S[:, :len(self.old_classes)],
                                                new_xs_outputs_T[:, :len(self.old_classes)].detach())
                new_data_CE_loss = F.cross_entropy(new_xs_outputs_S[:, len(self.old_classes):len(self.old_classes) + len(self.current_classes)],
                                                new_ys - len(self.old_classes), reduction="mean")
                # new_data_CE_loss = F.cross_entropy(new_xs_outputs_S[:, :len(self.old_classes)+len(self.current_classes)],
                #                                 new_ys, reduction="mean")
                loss_S = self.config.odce_ratio * old_data_CE_loss + self.config.odkd_ratio * old_data_KD_loss \
                     + self.config.ndce_ratio * new_data_CE_loss + self.config.ndkd_ratio * new_data_KD_loss

                loss_S.backward()
                optimizer_S.step()

                SO_CE_loss.update(old_data_CE_loss.item())
                SO_KD_loss.update(old_data_KD_loss.item())
                SN_CE_loss.update(new_data_CE_loss.item())
                SN_KD_loss.update(new_data_KD_loss.item())

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(new_xs), len(new_tasks_dataloader.dataset),
                #             100. * batch_idx / len(new_tasks_dataloader), new_data_loss.item()))

            if schedulers is not None:
                def step_scheduler(scheduler):
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        scheduler.step(epoch - 1 + batch_idx / self.batches_per_epoch)
                    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                if isinstance(schedulers, tuple):
                    for scheduler in schedulers:
                        step_scheduler(scheduler)
                else:
                    step_scheduler(schedulers)

            if (batch_idx + 1) % log_interval == 0:
                # print(alpha)
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
                      'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
                      'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
                      format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                             SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
            # self.test_old_and_new()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
              'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
              'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
              format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                     SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
        #self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg, SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg

    def train_dif_lr(self, epoch, new_tasks_dataloader, optimizer_G, optimizer_SO, optimizer_SN, log_interval=100, schedulers=None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()

        C_loss, A_loss, KD_loss = AverageMeter(), AverageMeter(), AverageMeter()
        correct_T, correct_S, total_T, total_S = 0, 0, 0, 0
        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            self.student.train()
            z = torch.randn([self.batch_size, self.noise_dim]).to(get_device(self.config.cuda))
            #GT_label = torch.randint(0, self.num_classes, [self.batch_size]).to(get_device(self.config.cuda))
            GT_label = torch.randint(0, len(self.old_classes), [self.batch_size]).to(get_device(self.config.cuda))

            new_xs = new_xs.to(get_device(self.config.cuda))
            new_ys = new_ys.to(get_device(self.config.cuda))

            optimizer_G.zero_grad()
            # gen_imgs = self.generator(z, GT_label)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
            gen_imgs = self.generator(z)
            outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)

            # generator
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_G:
                loss_activation = -features_T.abs().mean()
                loss_classification = self.LSR_loss(outputs_T[:, :len(self.old_classes)], GT_label)
                outputs_S = self.student(gen_imgs)
                loss_adv_student = -self.KD_loss(outputs_S[:, :len(self.old_classes)], outputs_T[:, :len(self.old_classes)])
                T_label = outputs_T[:, :len(self.old_classes)].data.max(1)[1]
                loss_onehot = self.CE_loss(outputs_T[:, :len(self.old_classes)], T_label)
                softmax_o_T = torch.nn.functional.softmax(outputs_T[:, :len(self.old_classes)], dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_G = loss_classification * self.ce_ratio + loss_onehot * self.oh_ratio + loss_information_entropy * self.ie_ratio + loss_activation * self.a_ratio + loss_adv_student * self.nkd_ratio
                loss_G.backward()
                optimizer_G.step()

                C_loss.update(loss_classification.item())
                A_loss.update(loss_activation.item())

                correct_T += (torch.argmax(outputs_T[:, :len(self.old_classes)], 1) == T_label).sum().item()
                total_T += GT_label.shape[0]

            # student update on new data
            if batch_idx % max(self.update_freq_G, self.update_freq_SN) < self.update_freq_SN:
                #print("update_S_batch_id{}".format(batch_idx))
                optimizer_SN.zero_grad()
                new_xs_outputs_S = self.student(new_xs)
                new_data_loss = F.cross_entropy(new_xs_outputs_S[:, len(self.old_classes):len(self.old_classes) + len(self.current_classes)],
                                                new_ys - len(self.old_classes), reduction="mean")
                # new_data_loss = F.cross_entropy(new_xs_outputs_S[:, :len(self.old_classes)+len(self.current_classes)],
                #                                 new_ys, reduction="mean")
                new_data_loss = new_data_loss
                new_data_loss.backward()
                optimizer_SN.step()

                # KD_loss.update(new_data_loss.item())

                correct_S += (torch.argmax(outputs_S, 1) == T_label).sum().item()
                total_S += T_label.shape[0]

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(new_xs), len(new_tasks_dataloader.dataset),
                #             100. * batch_idx / len(new_tasks_dataloader), new_data_loss.item()))

            # student update on old generate data
            if batch_idx % max(self.update_freq_G, self.update_freq_SO) < self.update_freq_SO:
                #print("update_S_batch_id{}".format(batch_idx))
                optimizer_SO.zero_grad()
                outputs_S = self.student(gen_imgs.detach())
                loss_S = self.KD_loss(outputs_S[:, :len(self.old_classes)], outputs_T[:, :len(self.old_classes)].detach())
                loss_S.backward()
                optimizer_SO.step()

                KD_loss.update(loss_S.item())

                correct_S += (torch.argmax(outputs_S, 1) == T_label).sum().item()
                total_S += T_label.shape[0]

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(new_xs), len(new_tasks_dataloader.dataset),
                #             100. * batch_idx / len(new_tasks_dataloader), new_data_loss.item()))


            if schedulers is not None:
                def step_scheduler(scheduler):
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        scheduler.step(epoch - 1 + batch_idx / self.batches_per_epoch)
                    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                if isinstance(schedulers, tuple):
                    for scheduler in schedulers:
                        step_scheduler(scheduler)
                else:
                    step_scheduler(schedulers)

            if (batch_idx + 1) % log_interval == 0:
                # print(alpha)
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, S_loss (KD): {:.3f}, '
                      'T_prec: {:.1f}%, S_prec: {:.1f}%, Time: {:.3f}'.
                      format(epoch, (batch_idx + 1), C_loss.avg, A_loss.avg, KD_loss.avg,
                             100. * correct_T / total_T, 100. * correct_S / total_S, t_epoch))
            self.test_old_and_new()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, S_loss (KD): {:.3f}, '
              'T_prec: {:.1f}%, S_prec: {:.1f}%, Time: {:.3f}'.
              format(epoch, self.batches_per_epoch, C_loss.avg, A_loss.avg, KD_loss.avg,
                     100. * correct_T / total_T, 100. * correct_S / total_S, t_epoch))
        #self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return C_loss.avg, A_loss.avg, KD_loss.avg, 100. * correct_T / total_T, 100. * correct_S / total_S

    def train_old_new_resp(self, epoch, new_tasks_dataloader, optimizer_G, optimizer_SO, optimizer_SN, log_interval=100, schedulers=None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()

        G_CE_loss, G_A_loss, G_AKD_loss, \
        SO_CE_loss, SO_KD_loss, \
        SN_CE_loss, SN_KD_loss = \
            AverageMeter(), AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter(), \
            AverageMeter(), AverageMeter()

        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            self.student.train()
            new_xs = new_xs.to(get_device(self.config.cuda))
            new_ys = new_ys.to(get_device(self.config.cuda))
            # student update on new data
            for i in range(self.update_freq_SN):
                optimizer_SN.zero_grad()
                new_xs_outputs_S = self.student(new_xs)
                # new_data_loss = F.cross_entropy(new_xs_outputs_S[:, len(self.old_classes):len(self.old_classes) + len(self.current_classes)],
                #                                 new_ys - len(self.old_classes), reduction="mean")
                new_data_CE_loss = F.cross_entropy(new_xs_outputs_S[:, :len(self.old_classes)+len(self.current_classes)],
                                                new_ys, reduction="mean")
                new_xs_outputs_T = self.teacher(new_xs, out_feature=False)
                new_data_KD_loss = self.KD_loss(new_xs_outputs_S[:, :len(self.old_classes)],
                                                new_xs_outputs_T[:, :len(self.old_classes)].detach())
                new_data_loss = new_data_CE_loss + 0*new_data_KD_loss
                new_data_loss.backward()
                optimizer_SN.step()
                SN_CE_loss.update(new_data_CE_loss.item())
                SN_KD_loss.update(new_data_KD_loss.item())
                # print('Train Epoch: {}, Batch:{}, SN_CE_loss: {:.3f},'.
                #       format(epoch, (batch_idx + 1), SN_CE_loss.asvg))
            # generator
            for i in range(self.update_freq_G):
                z = torch.randn([self.batch_size, self.noise_dim]).to(get_device(self.config.cuda))
                GT_label = torch.randint(0, len(self.old_classes), [self.batch_size]).to(get_device(self.config.cuda))
                optimizer_G.zero_grad()
                gen_imgs = self.generator(z)
                outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)
                loss_activation = -features_T.abs().mean()
                loss_classification = self.LSR_loss(outputs_T[:, :len(self.old_classes)], GT_label)
                outputs_S = self.student(gen_imgs)
                loss_adv_student = -self.KD_loss(outputs_S[:, :len(self.old_classes)], outputs_T[:, :len(self.old_classes)])
                T_label = outputs_T[:, :len(self.old_classes)].data.max(1)[1]
                loss_onehot = self.CE_loss(outputs_T[:, :len(self.old_classes)], T_label)
                softmax_o_T = torch.nn.functional.softmax(outputs_T[:, :len(self.old_classes)], dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_G = loss_classification * self.ce_ratio + loss_onehot * self.oh_ratio + loss_information_entropy * self.ie_ratio + loss_activation * self.a_ratio + loss_adv_student * self.nkd_ratio
                loss_G.backward()
                optimizer_G.step()

                G_CE_loss.update(loss_classification.item())
                G_A_loss.update(loss_activation.item())
                G_AKD_loss.update(loss_adv_student)

                # print('Train Epoch: {}, Batch:{}, G_AKD_loss: {:.3f},'.
                #       format(epoch, (batch_idx + 1), G_AKD_loss.avg))

            # student update on old generate data
            for i in range(self.update_freq_SO):
                optimizer_SO.zero_grad()
                old_xs_outputs_S = self.student(gen_imgs.detach())
                old_xs_outputs_T = self.teacher(gen_imgs.detach(), out_feature=False)
                old_xs_T_label = old_xs_outputs_T[:, :len(self.old_classes)].data.max(1)[1]
                old_data_CE_loss = self.CE_loss(old_xs_outputs_S[:, :len(self.old_classes) + len(self.current_classes)],
                                                old_xs_T_label)
                old_data_KD_loss = self.KD_loss(old_xs_outputs_S[:, :len(self.old_classes)], old_xs_outputs_T[:, :len(self.old_classes)].detach())
                old_data_loss = old_data_CE_loss + old_data_KD_loss
                old_data_loss.backward()
                optimizer_SO.step()

                SO_CE_loss.update(old_data_CE_loss.item())
                SO_KD_loss.update(old_data_KD_loss.item())

                # print('Train Epoch: {}, Batch:{}, SO_KD_loss: {:.3f},'.
                #       format(epoch, (batch_idx + 1), SO_KD_loss.avg))

            if schedulers is not None:
                def step_scheduler(scheduler):
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        scheduler.step(epoch - 1 + batch_idx / self.batches_per_epoch)
                    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                if isinstance(schedulers, tuple):
                    for scheduler in schedulers:
                        step_scheduler(scheduler)
                else:
                    step_scheduler(schedulers)

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
                      'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
                      'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
                      format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                             SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
            # self.test_old_and_new()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, G_AKD_loss: {:.3f},'
              'SO_loss (CE): {:.3f}, SO_loss (KD): {:.3f},'
              'SO_loss (CE): {:.3f}, SN_loss (KD): {:.3f},'.
              format(epoch, self.batches_per_epoch, G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg,
                     SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg))
        #self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return G_CE_loss.avg, G_A_loss.avg, G_AKD_loss.avg, SO_CE_loss.avg, SO_KD_loss.avg, SN_CE_loss.avg, SN_KD_loss.avg

    # This function input 3 models, teacher student new_tasks_model  optimizer_G, optimizer_S, optimizer_NT
    def train_stn(self, epoch, new_tasks_dataloader, optimizer_G, optimizer_S, optimizer_NT, log_interval = 100, schedulers = None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()
        self.new_tasks_model.train()

        C_loss, A_loss, KD_loss = AverageMeter(), AverageMeter(), AverageMeter()
        correct_T, correct_S, total_T, total_S = 0, 0, 0, 0
        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            z = torch.randn([self.batch_size, self.noise_dim]).to(get_device(self.config.cuda))
            # GT_label = torch.randint(0, self.num_classes, [self.batch_size]).to(get_device(self.config.cuda))
            GT_label = torch.randint(0, len(self.old_classes), [self.batch_size]).to(get_device(self.config.cuda))

            new_xs = new_xs.to(get_device(self.config.cuda))
            new_ys = new_ys.to(get_device(self.config.cuda))

            optimizer_G.zero_grad()
            # gen_imgs = self.generator(z, GT_label)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
            gen_imgs = self.generator(z)
            outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)

            # generator
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_G:
                # print("update_G_batch_id{}".format(batch_idx))
                # loss_activation = -features_T.abs().mean()
                # loss_classification = self.LSR_loss(outputs_T, GT_label)
                # outputs_S = self.student(gen_imgs)
                # loss_adv_student = -self.KD_loss(outputs_S, outputs_T)
                # T_label = outputs_T.data.max(1)[1]
                # loss_onehot = self.CE_loss(outputs_T, T_label)
                # softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
                # loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_activation = -features_T.abs().mean()
                loss_classification = self.LSR_loss(outputs_T[:, :len(self.old_classes)], GT_label)
                outputs_S = self.student(gen_imgs)
                loss_adv_student = -self.KD_loss(outputs_S[:, :len(self.old_classes)],
                                                 outputs_T[:, :len(self.old_classes)])
                T_label = outputs_T[:, :len(self.old_classes)].data.max(1)[1]
                loss_onehot = self.CE_loss(outputs_T[:, :len(self.old_classes)], T_label)
                softmax_o_T = torch.nn.functional.softmax(outputs_T[:, :len(self.old_classes)], dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_G = loss_classification * self.ce_ratio + loss_onehot * self.oh_ratio + loss_information_entropy * self.ie_ratio + loss_activation * self.a_ratio + loss_adv_student * self.nkd_ratio
                loss_G.backward()
                optimizer_G.step()

                C_loss.update(loss_classification.item())
                A_loss.update(loss_activation.item())

                correct_T += (torch.argmax(outputs_T[:, :len(self.old_classes)], 1) == T_label).sum().item()
                total_T += GT_label.shape[0]

            # student
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_S:
                # print("update_S_batch_id{}".format(batch_idx))
                optimizer_S.zero_grad()
                outputs_S = self.student(gen_imgs.detach())
                loss_S = self.KD_loss(outputs_S[:, :len(self.old_classes)],
                                      outputs_T[:, :len(self.old_classes)].detach())
                # + self.ce_ratio * self.LSR_loss(outputs_S, GT_label)
                # new_xs_outputs_S = self.student(new_xs)
                # new_data_loss_S = F.cross_entropy(new_xs_outputs_S[:, len(self.old_classes):],
                #                                 new_ys - len(self.old_classes), reduction="mean")
                # loss = loss_S + 0 * new_data_loss_S
                loss_S.backward()
                optimizer_S.step()

                # For new_tasks_model
                optimizer_NT.zero_grad()
                outputs_NT = self.new_tasks_model(gen_imgs.detach())
                old_data_loss_NT = self.KD_loss(outputs_NT[:, :len(self.old_classes)],
                                      outputs_T[:, :len(self.old_classes)].detach())
                new_xs_outputs_NT = self.new_tasks_model(new_xs)
                new_data_loss_NT = F.cross_entropy(new_xs_outputs_NT[:, len(self.old_classes):len(self.old_classes) + len(self.current_classes)],
                                                   new_ys - len(self.old_classes), reduction="mean")
                loss = old_data_loss_NT + 0.01*new_data_loss_NT
                loss.backward()
                optimizer_NT.step()

                KD_loss.update(loss_S.item())

                correct_S += (torch.argmax(outputs_S, 1) == T_label).sum().item()
                total_S += T_label.shape[0]

                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(new_xs), len(new_tasks_dataloader.dataset),
                #             100. * batch_idx / len(new_tasks_dataloader), new_data_loss.item()))

            if schedulers is not None:
                def step_scheduler(scheduler):
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        scheduler.step(epoch - 1 + batch_idx / self.batches_per_epoch)
                    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

                if isinstance(schedulers, tuple):
                    for scheduler in schedulers:
                        step_scheduler(scheduler)
                else:
                    step_scheduler(schedulers)

            if (batch_idx + 1) % log_interval == 0:
                # print(alpha)
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, S_loss (KD): {:.3f}, '
                      'T_prec: {:.1f}%, S_prec: {:.1f}%, Time: {:.3f}'.
                      format(epoch, (batch_idx + 1), C_loss.avg, A_loss.avg, KD_loss.avg,
                             100. * correct_T / total_T, 100. * correct_S / total_S, t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, G_loss (CE): {:.3f}, G_loss (L1): {:.3f}, S_loss (KD): {:.3f}, '
              'T_prec: {:.1f}%, S_prec: {:.1f}%, Time: {:.3f}'.
              format(epoch, self.batches_per_epoch, C_loss.avg, A_loss.avg, KD_loss.avg,
                     100. * correct_T / total_T, 100. * correct_S / total_S, t_epoch))
        # self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return C_loss.avg, A_loss.avg, KD_loss.avg, 100. * correct_T / total_T, 100. * correct_S / total_S

    def test_old_and_new(self):

        self.student.eval()

        acc_data = 0.
        counts = 0

        num_out = int(np.prod(self.config.task_out_dims))
        per_label_acc = np.zeros(num_out)
        per_label_counts = np.zeros(num_out)

        for x, y in self.test_loader:
            x = x.to(get_device(self.config.cuda))
            y = y.to(get_device(self.config.cuda))

            with torch.no_grad():
                preds = self.student(x)

            preds_flat = torch.argmax(preds, dim=1)

            acc_data += (preds_flat == y).sum().item()
            counts += y.shape[0]

            for c in range(num_out):
                pos = (y == c)
                per_label_acc[c] += (pos * (preds_flat == c)).sum().item()
                per_label_counts[c] += pos.sum().item()

        # acc over all data
        acc_data /= counts

        # acc per class
        per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
        per_label_acc /= per_label_counts

        # acc per seen task and avg
        acc = None
        last_classes = self.current_classes.cpu().numpy()
        seen_classes = self.old_classes.cpu().numpy()

        per_task_acc = defaultdict(list)
        for c in seen_classes:  # seen tasks only
            per_task_acc[self.config.class_dict_tasks[c]].append(per_label_acc[c])
        old_acc = 0.
        for task_i in per_task_acc:
            # assert (len(per_task_acc[task_i]) == config.classes_per_task)
            per_task_acc[task_i] = np.array(per_task_acc[task_i]).mean()
            old_acc += per_task_acc[task_i]
        old_acc /= len(per_task_acc)

        # compute last classes acc
        current_task_acc = defaultdict(list)
        for c in last_classes:  # seen tasks only
            current_task_acc[self.config.class_dict_tasks[c]].append(per_label_acc[c])
        current_acc = 0.
        for task_i in current_task_acc:
            # assert (len(current_task_acc[task_i]) == config.classes_per_task)
            current_task_acc[task_i] = np.array(current_task_acc[task_i]).mean()
            current_acc += current_task_acc[task_i]
        current_acc /= len(current_task_acc)
        print('acc over old tasks data {:.3f}, acc over current task data {:.3f}, acc over all data {:.3f}'
              .format(old_acc, current_acc, acc_data))

    def test(self, test_loader):
        self.student.eval()
        C_loss = AverageMeter()
        correct = 0
        t0 = time.time()
        for batch_idx, (img, label) in enumerate(test_loader):
            img, label = img.to(get_device(self.config.cuda)), label.to(get_device(self.config.cuda))
            with torch.no_grad():
                output = self.student(img)
            loss_classification = self.CE_loss(output, label)

            C_loss.update(loss_classification.item())

            correct += (torch.argmax(output, 1) == label).sum().item()

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, S_loss (CE): {:.3f}, S_prec: {:.1f}%, Time: {:.3f}'.
              format(C_loss.avg, 100. * correct / len(test_loader.dataset), t_epoch))

        return C_loss.avg, 100. * correct / len(test_loader.dataset)

    def sample_image(self, n_row, fname):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = torch.normal(0, 1, [n_row ** 2, self.noise_dim]).to(get_device(self.config.cuda))
        # Get label ranging from 0 to n_classes for n rows
        label = np.array([num for _ in range(n_row) for num in range(n_row)])
        label = torch.from_numpy(label).long().to(get_device(self.config.cuda))
        gen_imgs = self.generator(z)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
        save_image(gen_imgs.data, fname, nrow=n_row, normalize=True)
