import torchvision
import torch
from torchvision import transforms, datasets
import numpy as np
from .utils_dataset import split_images_labels
from .utils_dataset import merge_images_labels

def get_select_dataset(args):
    if args.dataset == 'cifar100':
        trainset, testset, evalset, X_train_total, Y_train_total, X_valid_total, Y_valid_total = get_cifar100(args)
    return trainset, testset, evalset, X_train_total, Y_train_total, X_valid_total, Y_valid_total


def get_cifar100(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=True,
                                             download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                            download=False, transform=transform_test)
    evalset = torchvision.datasets.CIFAR100(root=args.dataset_dir, train=False,
                                            download=False, transform=transform_test)

    X_train_total = np.array(trainset.data)
    Y_train_total = np.array(trainset.targets)
    X_valid_total = np.array(testset.data)  # test set is used as val set
    Y_valid_total = np.array(testset.targets)

    return trainset, testset, evalset, X_train_total, Y_train_total, X_valid_total, Y_valid_total


def get_cur_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total,Y_valid_total):
    if args.dataset == 'cifar100':
        if args.dataloader_type == 'il':
            trainloader, testloader = get_cifar100_cur_il_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total, Y_valid_total)
        elif args.dataloader_type == 'oracle':
            trainloader, testloader = get_cifar100_cur_oracle_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total,Y_valid_total)
        return trainloader, testloader


def get_cifar100_cur_il_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total,Y_valid_total):
    order_list = list(order)
    if args.nb_cl_fg == args.nb_cl:
        indices_train_subset = np.array([i in order[range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl)] for i in Y_train_total])
        indices_test_subset = np.array([i in order[range(0, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
    else: ### if args.nb_cl_fg != or >= args.nb_cl
        if iteration == 0:
            indices_train_subset = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_train_total])
            indices_test_subset = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        else:
            indices_train_subset = np.array([i in order[range(args.nb_cl_fg + (iteration-1) * args.nb_cl, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_train_total])
            indices_test_subset = np.array([i in order[range(0, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
    ### images
    X_train = X_train_total[indices_train_subset]
    X_valid = X_valid_total[indices_test_subset]
    ### labels
    Y_train = Y_train_total[indices_train_subset]
    Y_valid = Y_valid_total[indices_test_subset]
    print('Batch of classes number {0} arrives ...'.format(iteration + 1))
    ### transfer label to 0-19 20-39
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid = np.array([order_list.index(i) for i in Y_valid])
    ### put current il data to trainset
    trainset.data = X_train.astype('uint8')
    trainset.targets = map_Y_train
    testset.data = X_valid.astype('uint8')
    testset.targets = map_Y_valid
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid), max(map_Y_valid)))
    return trainloader, testloader

def get_cifar100_cur_oracle_dataloader(iteration, order, args, trainset, testset, X_train_total, Y_train_total, X_valid_total,Y_valid_total):
    order_list = list(order)
    if args.nb_cl_fg == args.nb_cl:
        indices_train_subset = np.array([i in order[range(0, (iteration + 1) * args.nb_cl)] for i in Y_train_total])
        indices_test_subset = np.array([i in order[range(0, (iteration + 1) * args.nb_cl)] for i in Y_valid_total])
    else: ### if args.nb_cl_fg != or >= args.nb_cl
        if iteration == 0:
            indices_train_subset = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_train_total])
            indices_test_subset = np.array([i in order[range(0, args.nb_cl_fg)] for i in Y_valid_total])
        else:
            indices_train_subset = np.array([i in order[range(0, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_train_total])
            indices_test_subset = np.array([i in order[range(0, args.nb_cl_fg + iteration * args.nb_cl)] for i in Y_valid_total])
    ### images
    X_train = X_train_total[indices_train_subset]
    X_valid = X_valid_total[indices_test_subset]
    ### labels
    Y_train = Y_train_total[indices_train_subset]
    Y_valid = Y_valid_total[indices_test_subset]
    print('Batch of classes number {0} arrives ...'.format(iteration + 1))
    ### transfer label to 0-19 20-39
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid = np.array([order_list.index(i) for i in Y_valid])
    ### put current il data to trainset
    trainset.data = X_train.astype('uint8')
    trainset.targets = map_Y_train
    testset.data = X_valid.astype('uint8')
    testset.targets = map_Y_valid
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid), max(map_Y_valid)))
    return trainloader, testloader

