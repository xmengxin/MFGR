import matplotlib.pyplot as plt


def draw_curve_og(path, train_loss, test_loss, train_prec, test_prec):
    fig = plt.figure()
    ax1 = fig.add_subplot(211, title="loss")
    ax2 = fig.add_subplot(212, title="prec")
    ax1.plot(train_loss, marker=',', label='train' + ': {:.3f}'.format(train_loss[-1]))
    ax1.plot(test_loss, marker='.', label='test' + ': {:.3f}'.format(test_loss[-1]))

    ax2.plot(train_prec, marker=',', label='train' + ': {:.1f}'.format(train_prec[-1]))
    ax2.plot(test_prec, marker='.', label='test' + ': {:.1f}'.format(test_prec[-1]))

    ax1.legend()
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)


def draw_curve_DFIL(path, train_loss_C_s, train_loss_A_s,train_loss_G_AKD_s,
                    train_loss_SO_CE_s, train_loss_SO_KD_s,
                    train_loss_SN_CE_s, train_loss_SN_KD_s,
                    test_old_tasks_acc_s, test_current_task_acc_s):
    fig = plt.figure()
    ax1 = fig.add_subplot(211, title="loss")
    ax2 = fig.add_subplot(212, title="prec")
    # ax1.plot(train_loss_C_s, marker=',', label='generator CE (train)' + ': {:.3f}'.format(train_loss_C_s[-1]))
    # ax1.plot(train_loss_A_s, marker='.', label='generator L1 (train)' + ': {:.3f}'.format(train_loss_A_s[-1]))
    # ax1.plot(train_loss_G_AKD_s, marker=',', label='generator AKD (train)' + ': {:.3f}'.format(train_loss_G_AKD_s[-1]))

    ax1.plot(train_loss_SO_CE_s, marker=',', label='student SO CE (train)' + ': {:.4f}'.format(train_loss_SO_CE_s[-1]))
    ax1.plot(train_loss_SO_KD_s, marker=',', label='student SO KD (train)' + ': {:.4f}'.format(train_loss_SO_KD_s[-1]))

    if len(train_loss_SN_CE_s) != 0 and len(train_loss_SN_KD_s) != 0:
        ax1.plot(train_loss_SN_CE_s, marker=',', label='student SN CE (train)' + ': {:.4f}'.format(train_loss_SN_CE_s[-1]))
        ax1.plot(train_loss_SN_KD_s, marker=',', label='student SN KD (train)' + ': {:.4f}'.format(train_loss_SN_KD_s[-1]))

    for key in test_old_tasks_acc_s:
        ax2.plot(test_old_tasks_acc_s[key], marker=',',
                 label='student (test_task{})'.format(key) + ': {:.4f}'.format(test_old_tasks_acc_s[key][-1]))
    for key in test_current_task_acc_s:
        ax2.plot(test_current_task_acc_s[key], marker=',',
                 label='student (test_task{})'.format(key) + ': {:.4f}'.format(test_current_task_acc_s[key][-1]))
    # ax2.plot(train_prec_T_s, marker=',', label='teacher (train)' + ': {:.1f}'.format(train_prec_T_s[-1]))
    # ax2.plot(train_prec_S_s, marker=',', label='student (train)' + ': {:.1f}'.format(train_prec_S_s[-1]))
    # ax2.plot(test_prec_s, marker='.', label='student (test)' + ': {:.1f}'.format(test_prec_s[-1]))

    ax1.legend()
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)


def draw_curve_SFDA(path, train_loss_G_content, train_loss_G_style, train_loss_G_entropy,
                    train_loss_S_C, train_loss_S_D, test_loss_S_C, test_tgt_acc, test_src_acc=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(311, title="loss")
    ax2 = fig.add_subplot(312, title="prec")
    ax3 = fig.add_subplot(313, title="prec")
    ax1.plot(train_loss_G_content, marker=',', label='generator content (train): {:.3f}'.format(train_loss_G_content[-1]))
    ax1.plot(train_loss_G_style, marker='.', label='generator style (train): {:.3f}'.format(train_loss_G_style[-1]))
    ax1.plot(train_loss_G_entropy, marker='.', label='generator entropy (train): {:.3f}'.format(train_loss_G_entropy[-1]))

    ax2.plot(train_loss_S_C, marker=',', label='student C (train): {:.3f}'.format(train_loss_S_C[-1]))
    ax2.plot(train_loss_S_D, marker=',', label='student D (train): {:.3f}'.format(train_loss_S_D[-1]))
    ax2.plot(test_loss_S_C, marker=',', label='student C (test): {:.3f}'.format(test_loss_S_C[-1]))

    if test_src_acc:
        ax3.plot(test_src_acc, marker='.', label='source (test): {:.1f}'.format(test_src_acc[-1]))
    ax3.plot(test_tgt_acc, marker='.', label='target (test): {:.1f}'.format(test_tgt_acc[-1]))

    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.savefig(path)
    plt.close(fig)


def draw_curve_DANN(path, train_C_losses, test_C_losses, train_D_losses,
                    train_MMD_losses, train_src_accs, test_tgt_accs, test_src_accs=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(211, title="loss")
    ax2 = fig.add_subplot(212, title="prec")
    ax1.plot(train_C_losses, marker=',', label='source C (train)' + ': {:.3f}'.format(train_C_losses[-1]))
    ax1.plot(test_C_losses, marker='.', label='target C (test)' + ': {:.3f}'.format(test_C_losses[-1]))

    ax1.plot(train_D_losses, marker=',', label='Discriminate (train)' + ': {:.3f}'.format(train_D_losses[-1]))
    ax1.plot(train_MMD_losses, marker=',', label='MMD (train)' + ': {:.3f}'.format(train_MMD_losses[-1]))

    ax2.plot(train_src_accs, marker=',', label='source (train)' + ': {:.1f}'.format(train_src_accs[-1]))
    if test_src_accs:
        ax2.plot(test_src_accs, marker='.', label='source (test)' + ': {:.1f}'.format(test_src_accs[-1]))
    ax2.plot(test_tgt_accs, marker='.', label='target (test)' + ': {:.1f}'.format(test_tgt_accs[-1]))

    ax1.legend()
    ax2.legend()
    fig.savefig(path)
    plt.close(fig)
