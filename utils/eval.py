from collections import OrderedDict, defaultdict

import numpy as np
import torch

from DFIL.util.general import get_device


# accs_data: average over all data (old metric)
# per_label_accs: acc per class
# per_task_accs: acc per task
# accs: average over all seen tasks (after last task, is same as Chaudry def.)
# forgetting: average over all seen tasks

def evaluate_basic(config, tasks_model, data_loader, is_val,
                   last_classes=None, seen_classes=None, tag=""):
    if is_val:
        prefix = "val"
    else:
        prefix = "test"
    tasks_model.eval()
    acc_data = 0.
    counts = 0
    num_out = int(np.prod(config.task_out_dims))
    per_label_acc = np.zeros(num_out)
    per_label_counts = np.zeros(num_out)

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(get_device(config.cuda))
            y = y.to(get_device(config.cuda))
            preds = tasks_model(x)
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
    last_classes = last_classes.cpu().numpy()
    seen_classes = seen_classes.cpu().numpy()

    per_task_acc = defaultdict(list)
    for c in seen_classes:  # seen tasks only
        per_task_acc[config.class_dict_tasks[c]].append(per_label_acc[c])
    old_acc = 0.
    for task_i in per_task_acc:
        # assert (len(per_task_acc[task_i]) == config.classes_per_task)
        per_task_acc[task_i] = np.array(per_task_acc[task_i]).mean()
        print('task_{} acc is {:.4f}'.format(task_i, per_task_acc[task_i]))
        old_acc += per_task_acc[task_i]
    old_acc /= len(per_task_acc)

    # compute last classes acc
    current_task_acc = defaultdict(list)
    for c in last_classes:  # seen tasks only
        current_task_acc[config.class_dict_tasks[c]].append(per_label_acc[c])
    current_acc = 0.
    for task_i in current_task_acc:
        # assert (len(current_task_acc[task_i]) == config.classes_per_task)
        current_task_acc[task_i] = np.array(current_task_acc[task_i]).mean()
        print('task_{} acc is {:.4f}'.format(task_i, current_task_acc[task_i]))
        current_acc += current_task_acc[task_i]
    current_acc /= len(current_task_acc)

    print('old tasks acc {:.4f}, current task data {:.4f}, all data acc {:.4f}'
          .format(old_acc, current_acc, acc_data))
    return per_task_acc, current_task_acc

def compute_forgetting(config, t, per_task_accs, last_classes):
    # per_task_acc is not equal length per timestep so can't array

    # assert (t % config.eval_freq == 0)

    # find task that just finished
    last_task_i = None
    for c in last_classes:
        task_i = config.class_dict_tasks[c]
        if last_task_i is None:
            last_task_i = task_i
        else:
            assert (last_task_i == task_i)

    forgetting_per_task = {}
    for task_i in range(
            last_task_i):  # excl last (tasks are numbered chronologically)
        best_acc = None
        for past_t in per_task_accs:
            if past_t == 0:
                continue  # not used
            if past_t == t:
                continue

            if task_i in per_task_accs[past_t]:
                if best_acc is None or per_task_accs[past_t][task_i] > best_acc:
                    best_acc = per_task_accs[past_t][task_i]
        assert (best_acc is not None)

        forgetting_per_task[task_i] = best_acc - per_task_accs[t][task_i]

    assert (len(forgetting_per_task) == last_task_i)
    return np.array(list(forgetting_per_task.values())).mean()
