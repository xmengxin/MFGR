import time
import torch
from .base_trainer import BaseTrainer
import numpy as np
from DFIL.util.general import get_device
from collections import OrderedDict, defaultdict
import sys


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, config, test_loader, old_classes, current_classes):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.old_classes = old_classes
        self.current_classes = current_classes
        self.test_loader = test_loader

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        sys.stdout.flush()
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            #loss = self.criterion(output, target)
            loss = self.criterion(output[:, len(self.old_classes):], target - len(self.old_classes))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))
            #self.test_old_and_new()
        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))



        return losses / len(data_loader), correct / (correct + miss)

    def test_old_and_new(self):


        self.model.eval()

        acc_data = 0.
        counts = 0

        num_out = int(np.prod(self.config.task_out_dims))
        per_label_acc = np.zeros(num_out)
        per_label_counts = np.zeros(num_out)

        for x, y in self.test_loader:
            x = x.to(get_device(self.config.cuda))
            y = y.to(get_device(self.config.cuda))

            with torch.no_grad():
                preds = self.model(x)

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
        sys.stdout.flush()
        print('acc over old tasks data {:.3f}, acc over current task data {:.3f}, acc over all data {:.3f}'
              .format(old_acc, current_acc, acc_data))

    def test(self, test_loader):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%'.format(losses / (len(test_loader) + 1),
                                                         100. * correct / (correct + miss)))

        return losses / len(test_loader), correct / (correct + miss)
