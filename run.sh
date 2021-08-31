CUDA_VISIBLE_DEVICES=0 python train_mfgr.py \
--dataset cifar100 \
--dataset_dir /home/xxm/datasets/cifar100/cifar-100-python \
--num_classes 100 \
--nb_cl_fg 20 --nb_cl 20 \
--base_lr 0.1 --scheduler cosWR --epochs 150 \
--model resnet34 \
--train_batch_size 128 \
--plot_cm \
--dataloader_type il \
--method dfkd \
--generator_type generator32 \
--epochs_G 500 \
--g_loss_type bn \
--gtv_ratio 10 \
--gbn_ratio 20 \
--c_loss_type gnkd_ncecut \
--loss_ratio_adaptive \
--tensorboard

