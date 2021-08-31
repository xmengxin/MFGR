from datasets.data import get_data
from .general import get_device, invert_dict
from DFIL.models import *

def get_model_and_data(config):
    config.task_in_dims = {"mnist5k": (28 * 28,), "miniimagenet": (3, 84, 84), "cifar10": (3, 32, 32), "cifar100": (3, 32, 32)}[config.data]
    config.task_out_dims = {"mnist5k": (10,), "miniimagenet": (100,), "cifar10": (10,), "cifar100": (100,)}[config.data]

    if config.data == 'cifar10':
        tasks_model = ResNet18().to(get_device(config.cuda))
    elif config.data == "cifar100":
        if config.model_arch == "resnet34":
            tasks_model = ResNet34(100).to(get_device(config.cuda))

    (tasks_trainloader, testloader, valloader), num_tasks, tasks_dict_classes, full_train_data = get_data(config)

    config.class_dict_tasks = invert_dict(full_train_data.task_dict_classes)

    return tasks_model, tasks_trainloader, testloader, valloader, num_tasks, tasks_dict_classes
