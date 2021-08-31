#!/usr/bin/env python
# coding=utf-8
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from utils.utils_pytorch import *
import matplotlib.pyplot as plt
from utils.general import plot_cm
import sys
sys.path.append("..")
from datasets.utils_dataset import merge_images_labels

def test_ac(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args):
    if args.dataset == 'cifar100':
        acc_old, acc_new, acc_total = test_cifar100(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args)
    elif args.dataset == 'tinyimagenet' or 'cub_imagenet':
        acc_old, acc_new, acc_total = test_imgstyle(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args)
    return acc_old, acc_new, acc_total

def test_cifar100(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args):
    tg_model.eval()
    order_list = list(order)
    if iteration == 0:
        indices_test_subset_old = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        indices_test_subset_new = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        start_new_class = 0
        end_new_class = args.nb_cl_fg
    else:
        if args.nb_cl_fg == args.nb_cl:
            indices_test_subset_old = np.array([i in order[range(0, iteration * args.nb_cl)] for i in Y_valid_total])
            indices_test_subset_new = np.array([i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
            start_new_class = iteration * args.nb_cl
            end_new_class = (iteration + 1) * args.nb_cl
        else:
            indices_test_subset_old = np.array([i in order[range(0, args.nb_cl_fg + (iteration-1) * args.nb_cl)] for i in Y_valid_total])
            indices_test_subset_new = np.array([i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
            start_new_class = args.nb_cl_fg + (iteration-1) * args.nb_cl
            end_new_class = args.nb_cl_fg + iteration * args.nb_cl
    ### compute old classes accuracy
    X_valid_old = X_valid_total[indices_test_subset_old]
    Y_valid_old = Y_valid_total[indices_test_subset_old]
    map_Y_valid_old = np.array([order_list.index(i) for i in Y_valid_old])
    evalset.data = X_valid_old.astype('uint8')
    evalset.targets = map_Y_valid_old
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,num_workers=2)
    acc_old = compute_accuracy_all_images(tg_model, evalloader)
    ### compute new classes accturacy
    X_valid_new = X_valid_total[indices_test_subset_new]
    Y_valid_new = Y_valid_total[indices_test_subset_new]
    map_Y_valid_new = np.array([order_list.index(i) for i in Y_valid_new])
    evalset.data = X_valid_new.astype('uint8')
    evalset.targets = map_Y_valid_new
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,num_workers=2)
    acc_new = compute_accuracy_all_images(tg_model, evalloader)
    ### compute total acc and each class acc
    acc_total, old_class_acc_mean, new_class_acc_mean, total_class_acc_mean = compute_accuracy_per_class(tg_model, testloader, start_new_class, end_new_class, args, iteration)
    print('Old images ac: {:.2f} % New images ac: {:.2f} % Total images ac: {:.2f} % '.format(acc_old, acc_new, acc_total))
    print('Old classes ac: {:.2f} % New classes ac: {:.2f} % Total classes ac: {:.2f} % '.format(old_class_acc_mean, new_class_acc_mean, total_class_acc_mean))
    return acc_old, acc_new, acc_total

def test_imgstyle(tg_model, X_valid_total, Y_valid_total, evalset, testloader, order, iteration, args):
    tg_model.eval()
    order_list = list(order)
    if iteration == 0:
        indices_test_subset_old = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        indices_test_subset_new = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        start_new_class = 0
        end_new_class = args.nb_cl_fg
    else:
        if args.nb_cl_fg == args.nb_cl:
            indices_test_subset_old = np.array([i in order[range(0, iteration * args.nb_cl)] for i in Y_valid_total])
            indices_test_subset_new = np.array([i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
            start_new_class = iteration * args.nb_cl
            end_new_class = (iteration + 1) * args.nb_cl
        else:
            indices_test_subset_old = np.array([i in order[range(0, args.nb_cl_fg + (iteration-1) * args.nb_cl)] for i in Y_valid_total])
            indices_test_subset_new = np.array([i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
            start_new_class = args.nb_cl_fg + (iteration-1) * args.nb_cl
            end_new_class = args.nb_cl_fg + iteration * args.nb_cl
    ### compute old classes accuracy
    X_valid_old = X_valid_total[indices_test_subset_old]
    Y_valid_old = Y_valid_total[indices_test_subset_old]
    map_Y_valid_old = np.array([order_list.index(i) for i in Y_valid_old])
    eval_set_old = merge_images_labels(X_valid_old, map_Y_valid_old)
    evalset.imgs = evalset.samples = eval_set_old
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,num_workers=2)
    acc_old = compute_accuracy_all_images(tg_model, evalloader)
    ### compute new classes accturacy
    X_valid_new = X_valid_total[indices_test_subset_new]
    Y_valid_new = Y_valid_total[indices_test_subset_new]
    map_Y_valid_new = np.array([order_list.index(i) for i in Y_valid_new])
    eval_set_new = merge_images_labels(X_valid_new, map_Y_valid_new)
    evalset.imgs = evalset.samples = eval_set_new
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,num_workers=2)
    acc_new = compute_accuracy_all_images(tg_model, evalloader)
    ### compute total acc and each class acc
    acc_total, old_class_acc_mean, new_class_acc_mean, total_class_acc_mean = compute_accuracy_per_class(tg_model, testloader, start_new_class, end_new_class, args, iteration)
    print('Old images ac: {:.2f} % New images ac: {:.2f} % Total images ac: {:.2f} % '.format(acc_old, acc_new, acc_total))
    print('Old classes ac: {:.2f} % New classes ac: {:.2f} % Total classes ac: {:.2f} % '.format(old_class_acc_mean, new_class_acc_mean, total_class_acc_mean))
    return acc_old, acc_new, acc_total

def compute_accuracy_all_images(tg_model, evalloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)
            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    cnn_acc = 100.*correct/total
    return cnn_acc

def compute_accuracy_per_class(tg_model, evalloader, start_new_class, end_new_class, args, task_id):
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    outputs_old_classes = []
    per_label_acc = np.zeros(end_new_class)
    per_label_counts = np.zeros(end_new_class)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            total += targets.size(0)
            all_targets.append(targets.cpu())
            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            outputs_old_classes.append(outputs[:,0:end_new_class].cpu())
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())
            for c in range(end_new_class):
                pos = (targets == c)
                per_label_acc[c] += (pos * (predicted == c)).sum().item()
                per_label_counts[c] += pos.sum().item()
    cnn_acc = 100.*correct/total
    per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
    per_label_acc /= per_label_counts
    total_class_acc_mean = 100.*per_label_acc.mean()
    if start_new_class == 0:
        old_class_acc_mean = 0
    else:
        old_class_acc_mean = 100. * per_label_acc[0:start_new_class].mean()
    new_class_acc_mean = 100. * per_label_acc[start_new_class:end_new_class].mean()
    ### plot confusion matrix
    if args.plot_cm:
        labels = np.array(list(range(1,end_new_class + 1)))
        plot_cm(np.concatenate(all_targets), np.concatenate(all_predicted), labels,
                vmax=50, title='Confusion matrix')
        save_cm_path = args.tensorboard_base_path + 'task{}'.format(task_id)
        os.makedirs(save_cm_path, exist_ok=True)
        plt.savefig(save_cm_path + '/confusion_matrix.jpg')

    ### save per class accuracy to excel
    import pandas as pd
    df = pd.DataFrame(per_label_acc)
    save_excel_path = args.tensorboard_base_path + 'task{}'.format(task_id)
    os.makedirs(save_excel_path, exist_ok=True)
    df.to_excel(save_excel_path +'/per_class_acc.xlsx', index=False)
    return cnn_acc, old_class_acc_mean, new_class_acc_mean, total_class_acc_mean

def test_cifar100_and_plot_cm(tg_model, X_valid_total, Y_valid_total, X_valid_ori, Y_valid_ori, evalset, testloader, order, order_list, iteration, args):
    tg_model.eval()
    # if iteration>start_iter:
    #     ## joint classifiers
    #     #num_old_classes = ref_model.fc.out_features
    #     tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
    #     tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
    print("##############################################################")
    # Calculate validation error of model on the original classes:
    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
    # print('Computing accuracy on the original batch of classes...')
    evalset.data = X_valid_ori.astype('uint8')
    evalset.targets = map_Y_valid_ori
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl_fg + (iteration-1) * args.nb_cl)
    print('Old classes accuracy: {:.2f} %'.format(acc_old))
    ##
    if iteration == 0:
        indices_test_subset_cur = np.array(
            [i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
    else:
        indices_test_subset_cur = np.array(
            [i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
    X_valid_cur = X_valid_total[indices_test_subset_cur]
    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
    # print('Computing accuracy on the original batch of classes...')
    evalset.data = X_valid_cur.astype('uint8')
    evalset.targets = map_Y_valid_cur
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl * (iteration + 1))
    print('New classes accuracy: {:.2f} %'.format(acc_cur))
    # Calculate validation error of model on the cumul of classes:
    acc = compute_accuracy_WI_and_plot_cm(tg_model, testloader, 0, args.nb_cl * (iteration + 1), args)
    print('Total accuracy: {:.2f} %'.format(acc))
    print("##############################################################")

    return acc_old, acc_cur, acc

def test_tiny_or_crossd_save_per_class_acc(tg_model, X_valid_total, Y_valid_total, X_valid_ori, Y_valid_ori, evalset, testloader, order, order_list, iteration, args):
    tg_model.eval()
    # if iteration>start_iter:
    #     ## joint classifiers
    #     #num_old_classes = ref_model.fc.out_features
    #     tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
    #     tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
    print("##############################################################")
    # Calculate validation error of model on the original classes:
    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
    # print('Computing accuracy on the original batch of classes...')
    ori_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
    evalset.imgs = evalset.samples = ori_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    if iteration == 0:
        acc_old, old_class_mean = compute_accuracy_WI_per_class_acc(tg_model, evalloader, 0, args.nb_cl_fg, args, iteration)
        print('Old classes accuracy: {:.2f} % old Per Class mean: {:.2f}%'.format(acc_old, old_class_mean))
    else:
        acc_old, old_class_mean = compute_accuracy_WI_per_class_acc(tg_model, evalloader, 0, args.nb_cl_fg + (iteration - 1) * args.nb_cl , args, iteration)
        print('Old classes accuracy: {:.2f} % old Per Class mean: {:.2f}%'.format(acc_old, old_class_mean))
        # acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl_fg + (iteration-1) * args.nb_cl)
        # print('Old classes accuracy: {:.2f} %'.format(acc_old))
    # indices_test_subset_cur = np.array(
    #     [i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
    if iteration == 0:
        indices_test_subset_cur = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
    else:
        indices_test_subset_cur = np.array([i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
    X_valid_cur = X_valid_total[indices_test_subset_cur]
    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
    current_eval_set = merge_images_labels(X_valid_cur, map_Y_valid_cur)
    evalset.imgs = evalset.samples = current_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    if iteration == 0:
        acc_cur, new_class_mean = compute_accuracy_WI_per_class_acc(tg_model, evalloader, 0, args.nb_cl_fg, args, iteration, new_class_per_acc=True)
        print('new accuracy: {:.2f} %  new per class mean: {:.2f}%'.format(acc_cur, new_class_mean))
    else:
        acc_cur, new_class_mean = compute_accuracy_WI_per_class_acc(tg_model, evalloader, args.nb_cl_fg, args.nb_cl_fg + args.nb_cl * iteration, args, iteration, new_class_per_acc=True)
        print('new accuracy: {:.2f} %  new per class mean: {:.2f}%'.format(acc_cur, new_class_mean))
    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl_fg + args.nb_cl * (iteration + 1))
    print('New classes accuracy: {:.2f} %'.format(acc_cur))
    # Calculate validation error of model on the cumul of classes:
    if iteration == 0:
        acc, class_mean = compute_accuracy_WI_per_class_acc(tg_model, testloader, 0, args.nb_cl_fg, args, iteration)
        print('Total accuracy: {:.2f} % Total per class mean: {:.2f}%'.format(acc, class_mean))
        print("##############################################################")
    else:
        acc, class_mean = compute_accuracy_WI_per_class_acc(tg_model, testloader, 0, args.nb_cl_fg + args.nb_cl * iteration, args, iteration)
        print('Total accuracy: {:.2f} % Total per class mean: {:.2f}%'.format(acc, class_mean))
        print("##############################################################")

    return acc_old, acc_cur, acc

def compute_accuracy_WI_per_class_acc(tg_model, evalloader, start_class, end_class, args, task_id, new_class_per_acc=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    outputs_old_classes = []
    per_label_acc = np.zeros(end_class)
    per_label_counts = np.zeros(end_class)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())
            #targets = targets - start_class
            outputs = tg_model(inputs)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            # outputs = F.softmax(outputs[:,0:end_class], dim=1)
            outputs_old_classes.append(outputs[:,0:end_class].cpu())


            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())

            for c in range(end_class):
                pos = (targets == c)
                per_label_acc[c] += (pos * (predicted == c)).sum().item()
                per_label_counts[c] += pos.sum().item()

    cnn_acc = 100.*correct/total
    per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
    per_label_acc /= per_label_counts
    class_acc_mean = 100.*per_label_acc.mean()
    per_label_acc_dict = {i: per_label_acc[i] for i in range(end_class)}
    # print (per_label_acc_dict)
    # print (class_acc_mean)

    import pandas as pd
    df = pd.DataFrame(per_label_acc)
    save_excel_path = args.tensorboard_base_path + 'task{}'.format(task_id)
    os.makedirs(save_excel_path, exist_ok=True)
    df.to_excel(save_excel_path +'/per_class_acc.xlsx', index=False)

    if new_class_per_acc:
        new_class_acc_mean = 100.*per_label_acc[start_class:end_class].mean()
        return cnn_acc, new_class_acc_mean
    else:
        return cnn_acc, class_acc_mean

def test_tiny_or_crossd(tg_model, X_valid_total, Y_valid_total, X_valid_ori, Y_valid_ori, evalset, testloader, order, order_list, iteration, args):
    tg_model.eval()
    # if iteration>start_iter:
    #     ## joint classifiers
    #     #num_old_classes = ref_model.fc.out_features
    #     tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
    #     tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
    print("##############################################################")
    # Calculate validation error of model on the original classes:
    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
    # print('Computing accuracy on the original batch of classes...')
    ori_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
    evalset.imgs = evalset.samples = ori_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl * (iteration + 1))
    print('Old classes accuracy: {:.2f} %'.format(acc_old))
    ##
    # indices_test_subset_cur = np.array(
    #     [i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
    if iteration == 0:
        indices_test_subset_cur = np.array(
            [i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
    else:
        indices_test_subset_cur = np.array(
            [i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
    X_valid_cur = X_valid_total[indices_test_subset_cur]
    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
    # print('Computing accuracy on the original batch of classes...')
    current_eval_set = merge_images_labels(X_valid_cur, map_Y_valid_cur)
    evalset.imgs = evalset.samples = current_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl * (iteration + 1))
    print('New classes accuracy: {:.2f} %'.format(acc_cur))
    # Calculate validation error of model on the cumul of classes:
    acc = compute_accuracy_WI(tg_model, testloader, 0, args.nb_cl * (iteration + 1))
    print('Total accuracy: {:.2f} %'.format(acc))
    print("##############################################################")

    return acc_old, acc_cur, acc

def test_tiny_or_crossd_oracle(tg_model, X_valid_total, Y_valid_total, X_valid_ori, Y_valid_ori, evalset, testloader, order, order_list, iteration, args):
    tg_model.eval()
    # if iteration>start_iter:
    #     ## joint classifiers
    #     #num_old_classes = ref_model.fc.out_features
    #     tg_model.fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
    #     tg_model.fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
    print("##############################################################")
    # Calculate validation error of model on the original classes:
    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
    # print('Computing accuracy on the original batch of classes...')
    ori_eval_set = merge_images_labels(X_valid_ori, map_Y_valid_ori)
    evalset.imgs = evalset.samples = ori_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_old = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl * (iteration + 1))
    print('Old classes accuracy: {:.2f} %'.format(acc_old))
    ##
    indices_test_subset_cur = np.array(
        [i in order[range(args.nb_cl_pre,args.nb_cl_fg)] for i in Y_valid_total])
    # indices_test_subset_cur = np.array(
    #     [i in order[range(60,80)] for i in Y_valid_total])
    X_valid_cur = X_valid_total[indices_test_subset_cur]
    Y_valid_cur = Y_valid_total[indices_test_subset_cur]
    map_Y_valid_cur = np.array([order_list.index(i) for i in Y_valid_cur])
    # print('Computing accuracy on the original batch of classes...')
    current_eval_set = merge_images_labels(X_valid_cur, map_Y_valid_cur)
    evalset.imgs = evalset.samples = current_eval_set
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=2)
    acc_cur = compute_accuracy_WI(tg_model, evalloader, 0, args.nb_cl * (iteration + 1))
    print('New classes accuracy: {:.2f} %'.format(acc_cur))
    # Calculate validation error of model on the cumul of classes:
    acc = compute_accuracy_WI(tg_model, testloader, 0, args.nb_cl * (iteration + 1))
    print('Total accuracy: {:.2f} %'.format(acc))
    print("##############################################################")

    return acc_old, acc_cur, acc


def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = np.squeeze(tg_feature_model(inputs)).cpu()
            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]


def compute_accuracy_CNN(tg_model, evalloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_WI_and_plot_cm(tg_model, evalloader, start_class, end_class, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())
            #targets = targets - start_class
            outputs = tg_model(inputs)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())

    cnn_acc = 100.*correct/total
    # cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))
    # df_cm = pd.DataFrame(cm, range(max(np.concatenate(all_targets)) + 1), range(max(np.concatenate(all_targets)) + 1))
    # # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
    # plt.savefig('1.jpg')
    # plt.show()

    # if args.load_task1_lwf:
    #     labels = np.array(list(range(60)))
    #     ax, sigma, sigma_norm = plot_cm(np.concatenate(all_targets), np.concatenate(all_predicted), labels,
    #                                     vmax=50, title='Confusion matrix')
    #     plt.savefig('cifar100_6tasks_task1_lwf.jpg')
    #     plt.show()
    if args.load_task1_ours:
        labels = np.array(list(range(60)))
        ax, sigma, sigma_norm = plot_cm(np.concatenate(all_targets), np.concatenate(all_predicted), labels,
                                        vmax=50, title='Confusion matrix')
        plt.savefig('cifar100_6tasks_task1_ours.jpg')
        plt.show()

    return cnn_acc

def compute_accuracy_WI(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []
    outputs_old_classes = []
    per_label_acc = np.zeros(60)
    per_label_counts = np.zeros(60)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())
            #targets = targets - start_class
            outputs = tg_model(inputs)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            # outputs_old_classes.append(outputs[:,0:50].cpu())


            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())

            # for c in range(60):
            #     pos = (targets == c)
            #     per_label_acc[c] += (pos * (predicted == c)).sum().item()
            #     per_label_counts[c] += pos.sum().item()

    cnn_acc = 100.*correct/total
    # per_label_counts = np.maximum(per_label_counts, 1)  # avoid div 0
    # per_label_acc /= per_label_counts


    # aa = np.concatenate(outputs_old_classes)
    # ab = np.mean(aa, axis=0)
    # print(np.argsort(ab))


    # cm = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))
    # df_cm = pd.DataFrame(cm, range(max(np.concatenate(all_targets)) + 1), range(max(np.concatenate(all_targets)) + 1))
    # # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
    # plt.savefig('1.jpg')
    # plt.show()

    # labels = np.array(list(range(100)))
    # ax, sigma, sigma_norm = plot_cm(np.concatenate(all_targets), np.concatenate(all_predicted), labels,
    #                                 vmax=50, title='Confusion matrix')
    # plt.savefig('test1.jpg')
    # plt.show()

    return cnn_acc

def compute_accuracy_Version1(tg_model, evalloader, nb_cl, nclassifier, iteration):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)
            #outputs = F.softmax(outputs, dim=1)
            real_classes = int(outputs.size(1)/nclassifier)
            nstep = iteration+1
            outputs_sum = torch.zeros(outputs.size(0), real_classes).to(device)
            ##
            for i in range(nstep):
                start = nb_cl*nclassifier*i
                for j in range(nclassifier):
                    end = start+nb_cl
                    outputs_sum[:, i*nb_cl:(i+1)*nb_cl] += outputs[:, start:end]
                    start = end

            # for i in range(nstep):
            #     start = nb_cl*nclassifier*i
            #     outputs_1 = F.softmax(outputs[:, start:start+nb_cl], dim=1)
            #     outputs_2 = F.softmax(outputs[:, start+nb_cl:start + 2*nb_cl], dim=1)
            #     ratio = torch.sum(torch.abs(outputs_1 - outputs_2), 1)
            #     outputs_sum[:, i*nb_cl:(i+1)*nb_cl] = outputs_1 #(outputs_1+outputs_2) * torch.unsqueeze(2.0 - ratio, 1)

            outputs_sum = F.softmax(outputs_sum, dim=1)
            _, predicted = outputs_sum.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100. * correct / total

    return cnn_acc

def compute_discrepancy(tg_model, evalloader, nb_cl, nclassifier, iteration, discrepancy):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    total = 0
    nstep = iteration + 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)
            ##
            for i in range(nstep):
                start_index = nb_cl*nclassifier*i
                for iter_1 in range(nclassifier):
                    outputs_1 = outputs[:, (start_index + nb_cl * iter_1):(start_index + nb_cl * (iter_1 + 1))]
                    outputs_1 = F.softmax(outputs_1, dim=1)
                    for iter_2 in range(iter_1 + 1, nclassifier):
                        outputs_2 = outputs[:, (start_index + nb_cl * iter_2):(start_index + nb_cl * (iter_2 + 1))]
                        outputs_2 = F.softmax(outputs_2, dim=1)
                        discrepancy[targets.size(0)*batch_idx:targets.size(0)*(batch_idx+1),i] += torch.sum(torch.abs(outputs_1 - outputs_2), 1)

    return discrepancy


def compute_accuracy_Side(tg_model, evalloader, nb_cl, nclassifier, iteration, inds):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, side_fc=True)

            batch_inds = inds[batch_idx*targets.size(0):(batch_idx+1)*targets.size(0)]

            real_classes = int(outputs.size(1)/nclassifier)
            nstep = iteration+1
            outputs_sum = torch.zeros(outputs.size(0), nb_cl).to(device)
            ##

            start = nb_cl*nclassifier*batch_inds
            for j in range(nclassifier):
                end = start+nb_cl
                outputs_sum += outputs[:, start:end]
                start = end

            outputs_sum = outputs_sum/nclassifier
            outputs_sum = F.softmax(outputs_sum, dim=1)
            _, predicted = outputs_sum.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100. * correct / total

    return cnn_acc


def compute_accuracy_Step1(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, cls_fc=True)
            # for i in range(args.num_cls):
            outputs_1 = outputs[:, :20]
            outputs_2 = outputs[:, 20:40]
            #
            outputs_1 = F.softmax(outputs_1, dim=1)
            _, predicted1 = outputs_1.max(1)
            correct1 += predicted1.eq(targets).sum().item()
            #
            outputs_2 = F.softmax(outputs_2, dim=1)
            _, predicted2 = outputs_2.max(1)
            correct2 += predicted2.eq(targets).sum().item()
            # fusion
            outputs_fusion = outputs[:, :20] + outputs[:, 20:40]
            _, predicted3 = outputs_fusion.max(1)
            correct3 += predicted3.eq(targets).sum().item()

    cnn_acc_1 = 100. * correct1 / total
    cnn_acc_2 = 100. * correct2 / total
    cnn_acc_3 = 100. * correct3 / total

    return cnn_acc_1, cnn_acc_2, cnn_acc_3

def compute_accuracy_Step2(tg_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct1 = 0
    correct2 = 0
    correct3 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            outputs = tg_model(inputs, cls_fc=True)
            #outputs = F.sigmoid(outputs)
            # for i in range(args.num_cls):
            old_outputs_1 = outputs[:, :20]
            old_outputs_2 = outputs[:, 20:40]
            old_outputs = (old_outputs_1 + old_outputs_2)/2
            #
            new_outputs_1 = outputs[:, 40:60]
            new_outputs_2 = outputs[:, 60:80]
            new_outputs = (new_outputs_1 + new_outputs_2) / 2
            ##
            final_outputs = torch.cat((old_outputs_1, new_outputs_1), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted1 = final_outputs.max(1)
            correct1 += predicted1.eq(targets).sum().item()

            final_outputs = torch.cat((old_outputs_2, new_outputs_2), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted2 = final_outputs.max(1)
            correct2 += predicted2.eq(targets).sum().item()

            final_outputs = torch.cat((old_outputs, new_outputs), dim=1)
            final_outputs = F.softmax(final_outputs, dim=1)
            _, predicted3 = final_outputs.max(1)
            correct3 += predicted3.eq(targets).sum().item()

    cnn_acc_1 = 100. * correct1 / total
    cnn_acc_2 = 100. * correct2 / total
    cnn_acc_3 = 100. * correct3 / total

    return cnn_acc_1, cnn_acc_2, cnn_acc_3

def compute_accuracy_AIG_Cls(tg_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            #targets = targets - start_class
            feats = tg_model(inputs, cls_fc=False)
            outputs = cls_model(feats)
            #outputs = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Semantic(tg_model, policy_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    correct_gates = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            total += targets.size(0)
            targets = targets - start_class
            inputs = inputs.to(device)
            targets = targets.to(device)

            gates, gates_cls = policy_model(inputs, temperature=temp)
            outputs = tg_model(inputs, gates)
            #outputs_sub = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            gates_cls = F.softmax(gates_cls, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            _, predicted_gates = gates_cls.max(1)
            correct_gates += predicted_gates.eq(targets).sum().item()

    cnn_acc = 100.*correct/total
    cnn_acc_gates = 100. * correct_gates / total

    return cnn_acc, cnn_acc_gates

def compute_accuracy_AIG_Semantic_Cls(tg_model, cls_model, policy_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    correct_gates = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            total += targets.size(0)
            targets = targets - start_class
            inputs = inputs.to(device)
            targets = targets.to(device)

            gates, gates_cls = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates, cls_fc=False)
            outputs = cls_model(feats)
            #outputs_sub = outputs[:, start_class: end_class]
            outputs = F.softmax(outputs, dim=1)
            gates_cls = F.softmax(gates_cls, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            _, predicted_gates = gates_cls.max(1)
            correct_gates += predicted_gates.eq(targets).sum().item()

    cnn_acc = 100.*correct/total
    cnn_acc_gates = 100. * correct_gates / total

    return cnn_acc, cnn_acc_gates


def compute_accuracy_Policy_Step1(tg_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = tg_model(inputs, gates=None, cls_fc=False)
            outputs = cls_model(feats)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Policy_Step1_Gated(tg_model, policy_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            new_gates = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates=new_gates, cls_fc=False)
            outputs = cls_model(feats)
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc


def compute_accuracy_Policy_Step2(tg_model, old_cls_model, new_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = tg_model(inputs, gates=None, cls_fc=False)
            old_logits = old_cls_model(feats)
            new_logits = new_cls_model(feats)
            logits = torch.cat((old_logits, new_logits), 1)
            logits = logits[:, start_class:end_class]
            logits = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_Policy_Step2_Gated(tg_model, policy_model, old_cls_model, new_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    temp = 1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class

            gates = policy_model(inputs, temperature=temp)
            feats = tg_model(inputs, gates=gates, cls_fc=False)

            old_logits = old_cls_model(feats)
            new_logits = new_cls_model(feats)
            logits = torch.cat((old_logits, new_logits), 1)
            logits = logits[:, start_class:end_class]
            logits = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Original(tg_model, evalloader, start_class, end_class, gates):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            if gates == True:
                outputs, _ = tg_model(inputs, temperature=1, openings=gates)
            else:
                outputs = tg_model(inputs, temperature=1, openings=gates)

            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs[:, start_class:end_class].max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_2(common_model, specific_model, cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            targets = targets - start_class
            feats = common_model(inputs, side=False)
            feats = specific_model(feats)
            logits = cls_model(feats)
            outputs = F.softmax(logits, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc

def compute_accuracy_AIG_Step2(common_model, task1_specific_model, task2_specific_model, task1_cls_model, task2_cls_model, evalloader, start_class, end_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #tg_feature_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            targets = targets - start_class

            feats = common_model(inputs, side=False)
            task1_feats = task1_specific_model(feats)
            task1_logits = task1_cls_model(task1_feats)
            task2_feats = task2_specific_model(feats)
            task2_logits = task2_cls_model(task2_feats)
            logits = torch.cat((task1_logits, task2_logits), 1)
            outputs = logits[:, start_class:end_class]
            outputs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


    #print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc


def compute_accuracy_without_FC(tg_model, evalloader, fc_cls, pool_classifers):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    fc_cls.eval()
    if len(pool_classifers)>0:
        for old_cls in pool_classifers:
            old_cls.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            probs = fc_cls(outputs)

            if len(pool_classifers)>0:
                for old_cls in reversed(pool_classifers):
                    old_probs = old_cls(outputs)
                    probs = torch.cat((old_probs, probs), 1)

            probs = F.softmax(probs, dim=1)
            #probs = F.sigmoid(probs)
            _, predicted = probs.max(1)
            correct += predicted.eq(targets).sum().item()


    print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))

    cnn_acc = 100.*correct/total

    return cnn_acc