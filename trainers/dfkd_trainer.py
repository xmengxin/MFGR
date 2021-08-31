import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from DFIL.util.meters import AverageMeter
from DFIL.util.general import *


class DFKDTrainer(BaseTrainer):
    def __init__(self, teacher, generator, student, CE_loss, LSR_loss, KD_loss, noise_dim, num_classes,
                 ratios, update_freqs, batch_size, batches_per_epoch, logdir=None):
        super(BaseTrainer, self).__init__()
        self.teacher = teacher
        self.generator = generator
        self.student = student
        self.CE_loss = CE_loss
        self.LSR_loss = LSR_loss
        self.KD_loss = KD_loss
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.a_ratio, self.oh_ratio, self.ie_ratio, self.ce_ratio, self.nkd_ratio, self.att_ratio = ratios
        self.update_freq_G, self.update_freq_S = update_freqs
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.logdir = logdir



    def train(self, epoch, new_tasks_dataloader, optimizer_G, optimizer_S, log_interval=100, schedulers=None):
        self.teacher.eval()
        self.generator.train()
        self.student.train()

        C_loss, A_loss, KD_loss = AverageMeter(), AverageMeter(), AverageMeter()
        correct_T, correct_S, total_T, total_S = 0, 0, 0, 0
        t0 = time.time()

        for batch_idx, (new_xs, new_ys) in enumerate(new_tasks_dataloader):
            z = torch.randn([self.batch_size, self.noise_dim]).cuda()
            GT_label = torch.randint(0, self.num_classes, [self.batch_size]).cuda()
            new_xs = new_xs.cuda()
            new_ys = new_ys.cuda()

            optimizer_G.zero_grad()
            # gen_imgs = self.generator(z, GT_label)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
            gen_imgs = self.generator(z)
            outputs_T, features_T = self.teacher(gen_imgs, out_feature=True)

            # generator
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_G:
                #print("update_G_batch_id{}".format(batch_idx))
                loss_activation = -features_T.abs().mean()
                loss_classification = self.LSR_loss(outputs_T, GT_label)
                outputs_S = self.student(gen_imgs)
                loss_adv_student = -self.KD_loss(outputs_S, outputs_T)
                T_label = outputs_T.data.max(1)[1]
                loss_onehot = self.CE_loss(outputs_T, T_label)
                softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
                loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                # loss_activation = -features_T.abs().mean()
                # loss_classification = self.LSR_loss(outputs_T[:, :20], GT_label)
                # outputs_S = self.student(gen_imgs)
                # loss_adv_student = -self.KD_loss(outputs_S[:, :20], outputs_T[:, :20])
                # T_label = outputs_T.data.max(1)[1]
                # loss_onehot = self.CE_loss(outputs_T, T_label)
                # softmax_o_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
                # loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()

                loss_G = loss_classification * self.ce_ratio + loss_onehot * self.oh_ratio + loss_information_entropy * self.ie_ratio + loss_activation * self.a_ratio + loss_adv_student * self.nkd_ratio
                loss_G.backward()
                optimizer_G.step()

                C_loss.update(loss_classification.item())
                A_loss.update(loss_activation.item())

                correct_T += (torch.argmax(outputs_T, 1) == T_label).sum().item()
                total_T += GT_label.shape[0]

            # student
            if batch_idx % max(self.update_freq_G, self.update_freq_S) < self.update_freq_S:
                #print("update_S_batch_id{}".format(batch_idx))
                optimizer_S.zero_grad()
                outputs_S = self.student(gen_imgs.detach())
                loss_S = self.KD_loss(outputs_S, outputs_T.detach())
                # + self.ce_ratio * self.LSR_loss(outputs_S, GT_label)
                # new_xs_outputs_S = self.student(new_xs)
                # new_data_loss = F.cross_entropy(new_xs_outputs_S, new_ys, reduction="mean")

                loss = loss_S
                # + 0.001*new_data_loss

                loss.backward()


                optimizer_S.step()

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
        #self.sample_image(n_row=10, fname=self.logdir + f'imgs/{epoch}.png')

        return C_loss.avg, A_loss.avg, KD_loss.avg, 100. * correct_T / total_T, 100. * correct_S / total_S

    def test(self, test_loader):
        self.student.eval()
        C_loss = AverageMeter()
        correct = 0
        t0 = time.time()
        for batch_idx, (img, label) in enumerate(test_loader):
            img, label = img.cuda(), label.cuda()
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
        z = torch.normal(0, 1, [n_row ** 2, self.noise_dim]).cuda()
        # Get label ranging from 0 to n_classes for n rows
        label = np.array([num for _ in range(n_row) for num in range(n_row)])
        label = torch.from_numpy(label).long().cuda()
        gen_imgs = self.generator(z)  # [:, 0].unsqueeze(1).repeat([1, 3, 1, 1])
        save_image(gen_imgs.data, fname, nrow=n_row, normalize=True)
