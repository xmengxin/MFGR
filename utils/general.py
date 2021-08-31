from __future__ import print_function
from __future__ import unicode_literals
import os.path as osp
import pickle
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib
from torch.autograd import Variable
matplotlib.use("agg")
import matplotlib.pyplot as plt
print(matplotlib.pyplot.get_backend())
from collections import OrderedDict, Counter
from copy import deepcopy
from matplotlib import pyplot as pyp
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime
from distutils.dir_util import copy_tree
import shutil
from utils.logger import Logger
import os
import sys
sys.path.append("..")

def set_random_seed(args, n_run):
  # seed
  if args.random_seed is not None:
    np.random.seed(args.random_seed + n_run + 1)
    torch.manual_seed(args.random_seed + n_run + 1)
    torch.cuda.manual_seed(args.random_seed + n_run + 1)
    # torch.cuda.manual_seed_all() #if you use multiple GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  else:
    torch.backends.cudnn.benchmark = True

def set_and_write_log(args):
  log_base_dir = f'./runs/{args.dataset}/{args.iteration_num}tasks/'
  logdir = log_base_dir + f'{args.model}_{args.method}_LT{args.c_loss_type}_ocncoknk{args.o_ce}{args.n_ce}{args.o_kd}{args.n_kd}_epochs_S{args.epochs}_Schedu{args.scheduler}_lrS{args.base_lr}_lrG{args.lr_G}' \
                          f'_oh{args.goh_ratio}_a{args.ga_ratio}_ie{args.gie_ratio}' \
                          f'_ntgb_{args.tn_batch_size_G}_b{args.train_batch_size}_' \
           + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + '/'
  copy_tree('data_process', logdir + '/scripts/data_process')
  copy_tree('datasets', logdir + '/scripts/datasets')
  copy_tree('losses', logdir + '/scripts/losses')
  copy_tree('models', logdir + '/scripts/models')
  copy_tree('plot_figure', logdir + '/scripts/plot_figure')
  copy_tree('trainers', logdir + '/scripts/trainers')
  copy_tree('utils', logdir + '/scripts/utils')
  for script in os.listdir('.'):
    if script.split('.')[-1] == 'py':
      dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
  sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )

  # set pretrained model path, tasks_model_path, curve imgs path, and tensorboard path
  args.teacher_model_path = log_base_dir
  args.tasks_model_path = logdir + 'tasks_model/'
  os.makedirs(args.tasks_model_path, exist_ok=True)
  if args.toy_G:
    args.tensorboard_base_path = log_base_dir + 'tensorboard_G/'
    os.makedirs(args.tensorboard_base_path, exist_ok=True)
  elif args.toy_example:
    args.tensorboard_base_path = log_base_dir + 'tensorboard_G/'
    os.makedirs(args.tensorboard_base_path, exist_ok=True)
  else:
    args.tensorboard_base_path = logdir + 'tensorboard/'
    os.makedirs(args.tensorboard_base_path, exist_ok=True)

  args.curve_imgs_path = logdir + 'imgs/'
  os.makedirs(args.curve_imgs_path, exist_ok=True)

def set_iteration_number(args):
  if args.nb_cl_fg == args.nb_cl:
    iteration_num = args.num_classes / args.nb_cl
  else:
    iteration_num = (args.num_classes - args.nb_cl_fg) / args.nb_cl + 1
  return iteration_num

def set_new_classifier(ref_model, tg_model, iteration, args):
  if args.model == 'resnet18_torchvision_model':
    if args.nb_cl_fg == args.nb_cl:
        num_old_classes = ref_model.fc.out_features
        in_features = ref_model.fc.in_features  # dim
        new_fc = torch.nn.Linear(in_features, args.nb_cl * (iteration + 1)).cuda()
        new_fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
        new_fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
        tg_model.fc = new_fc
        for param in tg_model.parameters():
          param.requires_grad = True
    else:
        num_old_classes = ref_model.fc.out_features
        in_features = ref_model.fc.in_features  # dim
        new_fc = torch.nn.Linear(in_features, args.nb_cl_fg + iteration * args.nb_cl).cuda()
        new_fc.weight.data[:num_old_classes] = ref_model.fc.weight.data
        new_fc.bias.data[:num_old_classes] = ref_model.fc.bias.data
        tg_model.fc = new_fc
        for param in tg_model.parameters():
          param.requires_grad = True
  else:
    if args.nb_cl_fg == args.nb_cl:
        num_old_classes = ref_model.linear.out_features
        in_features = ref_model.linear.in_features  # dim
        new_fc = torch.nn.Linear(in_features, args.nb_cl * (iteration + 1)).cuda()
        new_fc.weight.data[:num_old_classes] = ref_model.linear.weight.data
        new_fc.bias.data[:num_old_classes] = ref_model.linear.bias.data
        tg_model.linear = new_fc
        for param in tg_model.parameters():
          param.requires_grad = True
    else:
        num_old_classes = ref_model.linear.out_features
        in_features = ref_model.linear.in_features  # dim
        new_fc = torch.nn.Linear(in_features, args.nb_cl_fg + iteration * args.nb_cl).cuda()
        new_fc.weight.data[:num_old_classes] = ref_model.linear.weight.data
        new_fc.bias.data[:num_old_classes] = ref_model.linear.bias.data
        tg_model.linear = new_fc
        for param in tg_model.parameters():
          param.requires_grad = True
  return tg_model, num_old_classes

def set_targets_cut(args, targets, iteration):
  if args.nb_cl_fg == args.nb_cl:
    targets = targets - args.nb_cl * iteration
  else:
    targets = targets - (args.nb_cl_fg + (iteration - 1) * args.nb_cl)
  return targets

def fill_noise(x, noise_type):
  """Fills tensor `x` with noise of type `noise_type`."""
  if noise_type == 'u':
    x.uniform_()
  elif noise_type == 'n':
    x.normal_()
  else:
    assert False

def np_to_torch(img_np):
  '''Converts image in numpy.array to torch.Tensor.

  From C x W x H [0..1] to  C x W x H [0..1]
  '''
  return torch.from_numpy(img_np)[None, :]

def get_unet_noise_input(args,input_depth, method, spatial_size, noise_type='u', var=1. / 10):
  """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
  initialized in a specific way.
  Args:
      input_depth: number of channels in the tensor
      method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
      spatial_size: spatial size of the tensor to initialize
      noise_type: 'u' for uniform; 'n' for normal
      var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
  """
  if isinstance(spatial_size, int):
    spatial_size = (spatial_size, spatial_size)
  if method == 'noise':
    shape = [args.batch_size_G, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var
  elif method == 'meshgrid':
    assert input_depth == 2
    X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                       np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    net_input = np_to_torch(meshgrid)
  else:
    assert False

  return net_input

def get_generator_initial_noise(args):
  if args.generator_type == 'generator32':
    z = Variable(torch.randn(args.batch_size_G, args.latent_dim_G)).cuda()
  elif args.generator_type == 'generator128':
    z = Variable(torch.randn(args.batch_size_G, args.latent_dim_G)).cuda()
  elif args.generator_type == 'generator256':
    z = Variable(torch.randn(args.batch_size_G, args.latent_dim_G)).cuda()
  elif args.generator_type == 'unet':
    z = Variable(get_unet_noise_input(args, 3, 'noise', [224, 224]).type(torch.cuda.FloatTensor))
  return z

def get_class_number(old_gen_outputs):
  old_gen_outputs = F.softmax(old_gen_outputs, dim=1)
  old_gen_predicted = old_gen_outputs.max(1)
  aa = old_gen_predicted[1].cpu().numpy().tolist()
  a_result = Counter(aa)
  return a_result


def get_device(cuda):
  if cuda:
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def store(config):
  # anonymization
  data_path = config.data_path
  out_root = config.out_root
  out_dir = config.out_dir

  config.data_path = ""
  config.out_root = ""
  config.out_dir = ""

  with open(osp.join(out_dir, "config.pickle"),
            'wb') as outfile:
    pickle.dump(config, outfile)

  with open(osp.join(out_dir, "config.txt"),
            "w") as text_file:
    text_file.write("%s" % config)

  config.data_path = data_path
  config.out_root = out_root
  config.out_dir = out_dir


def get_avg_grads(model):
  total = None
  count = 0
  for p in model.parameters():
    sz = np.prod(p.grad.shape)
    grad_sum = p.grad.abs().sum()
    if total is None:
      total = grad_sum
      count = sz
    else:
      total += grad_sum
      count += sz

  return total / float(count)


def record_and_check(config, name, val, t):
  if hasattr(val, "item"):
    val = val.item()

  record(config, name, val, t)
  if not np.isfinite(val):
    print("value (probably loss) not finite, aborting:")
    print(name)
    print("t %d" % t)
    print(val)
    store(config)  # to store nan values
    exit(1)


def check(config, val, t):
  if not np.isfinite(val):
    print("value (probably loss) not finite, aborting:")
    print("t %d" % t)
    print(val)
    store(config)  # to store nan values
    exit(1)


def record(config, val_name, val, t, abs=False):
  if not hasattr(config, val_name):
    setattr(config, val_name, OrderedDict())

  storage = getattr(config, val_name)

  if "torch" in str(val.__class__):
    if abs:
      val = torch.abs(val)

    if val.dtype == torch.int64:
      assert (val.shape == torch.Size([]))
    else:
      val = torch.mean(val)

    storage[t] = val.item()  # either scalar loss or vector of grads
  else:
    if abs:
      val = abs(val)  # default abs

    storage[t] = val

  if not hasattr(config, "record_names"):
    config.record_names = []

  if not val_name in config.record_names:
    config.record_names.append(val_name)


def get_gpu_mem(nvsmi, gpu_ind):
  mem_stats = nvsmi.DeviceQuery('memory.free, memory.total')["gpu"][gpu_ind]["fb_memory_usage"]
  return mem_stats["total"] - mem_stats["free"]


def get_task_in_dims(config):
  return config.task_in_dims


def get_task_out_dims(config):
  return config.task_out_dims


def render_graphs(config):
  if not hasattr(config, "record_names"):
    return

  training_val_names = config.record_names
  fig0, axarr0 = plt.subplots(max(len(training_val_names), 2), sharex=False,
                              figsize=(8, len(training_val_names) * 4))

  for i, val_name in enumerate(training_val_names):
    if hasattr(config, val_name):
      storage = getattr(config, val_name)
      axarr0[i].clear()
      axarr0[i].plot(list(storage.keys()), list(storage.values()))  # ordereddict

      axarr0[i].set_title(val_name)

  fig0.suptitle("Model %d" % (config.model_ind), fontsize=8)
  fig0.savefig(osp.join(config.out_dir, "plots_0.png"))

  if hasattr(config, "val_accs"):
    fig1, axarr1 = plt.subplots(4, sharex=False, figsize=(8, 4 * 4))

    for pi, prefix in enumerate(["val", "test"]):
      accs_name = "%s_accs" % prefix
      axarr1[pi * 2].clear()
      axarr1[pi * 2].plot(list(getattr(config, accs_name).keys()),
                          list(getattr(config, accs_name).values()))  # ordereddict
      axarr1[pi * 2].set_title(accs_name)

      per_label_accs_name = "%s_per_label_accs" % prefix
      axarr1[pi * 2 + 1].clear()
      per_label_accs_t = getattr(config, per_label_accs_name).keys()
      per_label_accs_np = np.array(list(getattr(config, per_label_accs_name).values()))
      for c in range(int(np.prod(get_task_out_dims(config)))):
        axarr1[pi * 2 + 1].plot(list(per_label_accs_t), list(per_label_accs_np[:, c]), label=str(c))
      axarr1[pi * 2 + 1].legend()
      axarr1[pi * 2 + 1].set_title(per_label_accs_name)

    fig1.suptitle("Model %d" % (config.model_ind), fontsize=8)
    fig1.savefig(osp.join(config.out_dir, "plots_1.png"))

  # render predictions, if exist
  if hasattr(config, "aux_y_probs"):
    # time along x axis, classes along y axis
    fig2, ax2 = plt.subplots(1, figsize=(16, 8))  # width, height

    num_t = len(config.aux_y_probs)
    num_classes = int(np.prod(get_task_out_dims(config)))

    aux_y_probs = list(config.aux_y_probs.values())
    aux_y_probs = [aux_y_prob.numpy() for aux_y_prob in aux_y_probs]
    aux_y_probs = np.array(aux_y_probs)

    # print(aux_y_probs.shape)
    assert (aux_y_probs.shape == (len(config.aux_y_probs), int(np.prod(get_task_out_dims(config)))))

    aux_y_probs = aux_y_probs.transpose()  # now num classes, time
    min_val = aux_y_probs.min()
    max_val = aux_y_probs.max()

    # tile along y axis to make each class fatter. Should be same number of pixels altogether as
    # current t / 2
    scale = int(0.5 * float(num_t) / num_classes)
    if scale > 1:
      aux_y_probs = np.repeat(aux_y_probs, scale, axis=0)
      ax2.set_yticks(np.arange(num_classes) * scale)
      ax2.set_yticklabels(np.arange(num_classes))

    num_thousands = int(num_t / 1000)
    ax2.set_xticks(np.arange(num_thousands) * 1000)
    ax2.set_xticklabels(np.arange(num_thousands) * 1000 + list(config.aux_y_probs.keys())[0])

    im = ax2.imshow(aux_y_probs)
    fig2.colorbar(im, ax=ax2)
    # ax2.colorbar()

    fig2.suptitle("Model %d, max %f min %f" % (config.model_ind, max_val, min_val), fontsize=8)
    fig2.savefig(osp.join(config.out_dir, "plots_2.png"))

  plt.close("all")


def trim_config(config, next_t):
  # trim everything down to next_t numbers
  # we are starting at top of loop *before* eval step

  for val_name in config.record_names:
    storage = getattr(config, val_name)
    if isinstance(storage, list):
      assert (len(storage) >= (next_t))
      setattr(config, val_name, storage[:next_t])
    else:
      assert (isinstance(storage, OrderedDict))
      storage_copy = deepcopy(storage)
      for k, v in storage.items():
        if k >= next_t:
          del storage_copy[k]
      setattr(config, val_name, storage_copy)

  for prefix in ["val", "test"]:
    accs_storage = getattr(config, "%s_accs" % prefix)
    per_label_accs_storage = getattr(config, "%s_per_label_accs" % prefix)

    if isinstance(accs_storage, list):
      assert (isinstance(per_label_accs_storage, list))
      assert (len(accs_storage) >= (next_t) and len(per_label_accs_storage) >= (
        next_t))  # at least next_t stored

      setattr(config, "%s_accs" % prefix, accs_storage[:next_t])
      setattr(config, "%s_per_label_accs" % prefix, per_label_accs_storage[:next_t])
    else:
      assert (
        isinstance(accs_storage, OrderedDict) and isinstance(per_label_accs_storage, OrderedDict))
      for dn, d in [("accs", accs_storage), ("per_label_accs", per_label_accs_storage)]:
        d_copy = deepcopy(d)
        for k, v in d.items():
          if k >= next_t:
            del d_copy[k]
        setattr(config, dn, d_copy)

  # deal with window
  if config.long_window:
    # find index of first historical t for update >= next_t
    # set config.next_update_old_model_t = that t
    # trim history behind it, backing onto nest_t

    next_t_i = None
    for i, update_t in enumerate(config.next_update_old_model_t_history):
      if update_t > next_t:
        next_t_i = i
        break

    # there must be a t in update history that is greater than next_t
    # unless config.next_update_old_model_t >= next_t and we stopped before it was added to history
    # in which case we don't need to trim any update history

    if next_t_i is None:
      print("no trimming:")
      print(("config.next_update_old_model_t", config.next_update_old_model_t))
      print(("next_t", next_t))
      assert (config.next_update_old_model_t >= next_t)
    else:
      config.next_update_old_model_t = config.next_update_old_model_t_history[next_t_i]
      config.next_update_old_model_t_history = config.next_update_old_model_t_history[:next_t_i]


def sum_seq(seq):
  res = None
  for elem in seq:
    if res is None:
      res = elem
    else:
      res += elem
  return res


def np_rand_seed():  # fixed classes shuffling
  return 111


def reproc_settings(config):
  np.random.seed(0)  # set separately when shuffling data too
  if config.specific_torch_seed:
    torch.manual_seed(config.torch_seed)
  else:
    torch.manual_seed(config.model_ind)  # allow initialisations different per model

  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def copy_parameter_values(from_model, to_model):
  to_params = list(to_model.named_parameters())
  assert (isinstance(to_params[0], tuple) and len(to_params[0]) == 2)

  to_params = dict(to_params)

  for n, p in from_model.named_parameters():
    to_params[n].data.copy_(p.data)  # not clone


def make_valid_from_train(dataset, cut):
  tr_ds, val_ds = [], []
  for task_ds in dataset:
    x_t, y_t = task_ds

    # shuffle before splitting
    perm = torch.randperm(len(x_t))
    x_t, y_t = x_t[perm], y_t[perm]

    split = int(len(x_t) * cut)
    x_tr, y_tr = x_t[:split], y_t[:split]
    x_val, y_val = x_t[split:], y_t[split:]

    tr_ds += [(x_tr, y_tr)]
    val_ds += [(x_val, y_val)]

  return tr_ds, val_ds


def invert_dict(dict_to_invert):
  new_dict = {}
  for k, vs in dict_to_invert.items():
    for v in vs:
      new_dict[v] = k
  return new_dict

def plot_cm(y_true, y_pred, labels, norm_axis=1, cmap='jet', vmin=0,
            vmax=None, lognorm=False, xlabel='True class',
            ylabel='Predicted class', title=None):
  """Plot confusion matrix.
  Parameters
  ----------
  y_true : ndarray, (n_samples,)
      True classes.
  y_pred : ndarray, (n_samples,)
      Predicted classes.
  labels : list of str
      List of class labels, in order that they should be displayed in
      confusion matrix.
  norm_axis : int, optional
      Normalize confusion matrix to sum to one along specified axis. Relevant for text plotted inside cells.
      (Default: 1)
  cmap : str or matplotlib.colors.Colormap, optional
      Colormap to use.
      (Default: 'Reds')
  vmin : int, optional
      Lower unnormalized count cutoff for colormap.
      (Default: 0)
  vmax : int, optional
      Upper unnormalized count cutoff for colormap. If None, then inferred
      from data.
      (Default: None)
  lognorm : bool, optional
     If True, use log-scale for colormap.
     (Default: False)
  xlabel : str, optional
      Label for x-axis.
      (Default: 'True class')
  ylabel : str, optional
      Label for y-axis.
      (Default: 'Predicted class')
  title : str, optional
      Title for plot.
      (Default: None)
  Returns
  -------
  ax : matplotlib.pyplot.Axes
      Axes.
  sigma : ndarray, (n_classes, n_classes)
      Unnormalized confusion matrix. ``sigma[i, j]`` is the number of times
      true label ``labels[i]`` was predicted to be label ``labels[j]``.
  sigma_norm : ndarray, (n_classes, n_classes)
      Normalized confusion matrix. Equivalent to ``sigma`` normalized so that
      rows (``norm_axis=1``) or columns (``norm_axis=0``) sum to 1.
  """
  ASPECT = 10.  # Ratio of x-axis to y-axis.
  EPS = 1e-5
  cmap = cm.jet if cmap == 'jet' else cmap  # Protect against seaborn's hate
  # for jet.

  # Compute confusion matrix.
  sigma = confusion_matrix(y_true, y_pred, labels=labels)
  if norm_axis in [0, 1]:
    marginals = sigma.sum(axis=norm_axis, dtype='float64')
    sigma_norm = sigma / np.expand_dims(marginals, axis=norm_axis)
  else:
    raise ValueError('norm_axis must be one of {0, 1}. Got %d' % norm_axis)
  sigma = sigma + EPS
  sigma_norm = sigma_norm + EPS

  # Plot raw counts.
  vmin = max(vmin, EPS)
  kwargs = {'vmin': max(vmin, EPS),
            'vmax': vmax,
            }
  if lognorm:
    kwargs['norm'] = LogNorm()
  ax = sns.heatmap(sigma, xticklabels=labels, yticklabels=labels,
                   cmap=cmap, robust=True, square=True, cbar=False)
  ### show labels on x and y axis
  xticks = ax.xaxis.get_major_ticks()
  for i in range(len(xticks)):
    if i == 0:
      xticks[i].set_visible(True)
    elif (i + 1) % 10 == 0:
      xticks[i].set_visible(True)
    else:
      xticks[i].set_visible(False)
  yticks = ax.yaxis.get_major_ticks()
  for i in range(len(yticks)):
    if i == 0:
      yticks[i].set_visible(True)
    elif (i + 1) % 10 == 0:
      yticks[i].set_visible(True)
    else:
      yticks[i].set_visible(False)
  ax.set_ylabel(xlabel)
  ax.set_xlabel(ylabel)
  if title is not None:
    ax.set_title(title)
  pyp.tight_layout()
  return ax, sigma, sigma_norm
