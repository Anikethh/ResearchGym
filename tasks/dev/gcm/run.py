import numpy as np
from model import *
from data_loader import DataLoader
from config import config

def train_step(x, r, y, model, lr = 0.001, reg_ratio = 1e-6, use_adam=True):
    vars = model.variables
    with tf.GradientTape() as t:
        l2_reg = sum([tf.reduce_sum(tf.square(var)) for var in vars])
        y_pred, loss = model([x, r], y=y, is_test=lr==0)
        total_loss = loss + l2_reg * reg_ratio
        if lr > 0:
            if use_adam:
                opt = tf.keras.optimizers.legacy.Adam(lr)
                grads = t.gradient(total_loss, vars)
                grads_and_vars = [(tf.clip_by_value(j, -1, 1), i) for i, j in zip(vars, grads) if j is not None]
                opt.apply_gradients(grads_and_vars)
            else:
                grads = t.gradient(total_loss, vars)
                grads = [tf.convert_to_tensor(i) for i in grads]
                for grad, var in zip(grads, vars):
                    var.assign_sub(lr * grad)
    if lr==0:
        pass
    return y_pred, loss

def find_opt_threshold(y, y_pred):
    tmp = sorted(zip(y_pred, y), key=lambda x: x[0])
    tmp1 = np.cumsum([1-i[1] for i in tmp]).tolist()
    tmp2 = np.argmax([j * 2 - i for i,j in enumerate(tmp1)])
    return tmp[tmp2][0]

def get_auc(y, y_pred, threshold=0.5):
    tmp = sorted(zip(y_pred, y), key=lambda x: x[0])
    n, m, l, t = 0, 0, 0, 0
    for j, (r, k) in enumerate(tmp):
        if k == 0:
            n += 1
        else:
            m += n
        if r < threshold:
            if k == 0:
                l += 1
        else:
            if k == 1:
                l += 1
        t += (r - k) ** 2
    auc = m / (n * (len(y) - n) + 1e-5)
    acc = l / len(y)
    rms = (t / len(y)) ** 0.5
    return auc, acc, rms

data_loader = DataLoader()
def run_exp(dataset, seed, write_log):
    x_train, r_train, y_train, x_test, r_test, y_test, dense_dim, r_dim, is_binary = data_loader.load(dataset)
    y_train_mean = sum(y_train) / y_train.shape[0]
    print("*"*100 + f"\n{dataset}, train sample Num: {x_train.shape[0]}, mean y: {y_train_mean:.6f}, x dim: {dense_dim}, r dim: {r_dim}\n" + "*"*100)
    N = x_train.shape[0]
    conf = config[dataset]
    use_adam = conf["use_adam"]
    max_epoch = conf["max_epoch"]
    batch_size = conf["batch_size"]
    hidden_dim = conf["hidden_dim"]
    lr = conf["lr"]
    decay_r = conf["decay_r"]
    model_list = [
        MlpModel(name=f'MLP_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        PosNNModel(name=f'PosNN_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        MinmaxModel(name=f'MM_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        SmoothedMinmaxModel(name=f'SMM_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        ConstrainedModel(name=f'CMNN_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        ScalableModel(name=f'SMNN_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        HintModel(name=f'Hint_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
        PwlModel(name=f'PWL_{seed}', dense_dim=dense_dim, r_dim=r_dim, hidden_dim=hidden_dim, is_binary=is_binary),
    ]
    train_loss_history_list = [[] for _ in model_list]
    train_auc_history_list = [[] for _ in model_list]
    train_acc_history_list = [[] for _ in model_list]
    train_rms_history_list = [[] for _ in model_list]
    epoch = 0
    rnd_id = list(range(N))
    while epoch <= max_epoch:
        epoch += 1
        lr *= decay_r
        begin = 0
        np.random.seed(seed + epoch * 1000)
        np.random.shuffle(rnd_id)
        x_train = x_train[rnd_id]
        r_train = r_train[rnd_id]
        y_train = y_train[rnd_id]
        # train
        while begin < N:
            x = x_train[begin: batch_size + begin]
            r = r_train[begin: batch_size + begin]
            y = y_train[begin: batch_size + begin]
            begin += batch_size
            for model in model_list:
                _, _ = train_step(x, r, y, model, lr, use_adam=use_adam)
            info = f'{dataset}-{seed} process: {epoch}/{max_epoch} lr: {lr:.4f}\nmodel     \tloss    \trms     \tauc     \tacc     \n'
        # validate
        for model, loss_history, auc_history, acc_history, rms_history in (
                zip(model_list, train_loss_history_list, train_auc_history_list, train_acc_history_list,
                    train_rms_history_list)):
            m = y_test.shape[0] // 4
            y_pred1, loss1 = train_step(x_test[:m], r_test[:m], y_test[:m], model, 0)
            y_pred2, loss2 = train_step(x_test[m:2*m], r_test[m:2*m], y_test[m:2*m], model, 0)
            y_pred3, loss3 = train_step(x_test[2*m:3*m], r_test[2*m:3*m], y_test[2*m:3*m], model, 0)
            y_pred4, loss4 = train_step(x_test[3*m:], r_test[3*m:], y_test[3*m:], model, 0)
            y_pred = tf.concat([y_pred1, y_pred2, y_pred3, y_pred4], 0)
            y_pred = tf.reshape(y_pred, [-1]).numpy().tolist()
            loss = (loss1+loss2+loss3+loss4).numpy().tolist() / 4
            loss_history.append(loss)
            threshold = 0.5
            auc, acc, rms = get_auc(tf.reshape(y_test, [-1]).numpy().tolist(), y_pred, threshold)
            if not is_binary:
                auc, acc = 0, 0
            if dataset == 'auto':
                rms *= 10
            auc_history.append(auc)
            acc_history.append(acc)
            rms_history.append(rms)
            n = model.name
            n = '_'.join(n.split('_')[:-1])
            info += f'{n+" "*(10-len(n))}\t{loss:.6f}\t{rms:.6f}\t{auc:.6f}\t{acc:.6f}\n'
        print(info[:-1], flush=True)
    print('Final result', flush=True)
    info = 'model     \tloss    \trms     \tauc     \tacc     \tbest step\n'
    #
    for model, l1, l2, l3, l4 in zip(model_list,
                                     train_loss_history_list,
                                     train_rms_history_list,
                                     train_auc_history_list,
                                     train_acc_history_list):
        n = model.name
        n = '_'.join(n.split('_')[:-1])
        ealy_stop_metric = [-i for i in l2] if not is_binary else (l3 if abs(y_train_mean - 0.5)>0.2 else l4)
        best_step = np.argmax(ealy_stop_metric)
        info += f'{n + " " * (10 - len(n))}\t{l1[best_step]:.6f}\t{l2[best_step]:.6f}\t{l3[best_step]:.6f}\t{l4[best_step]:.6f}\t{best_step}\n'
    print(info, flush=True)
    if write_log:
        with open(f'log/{dataset}_{seed}', 'w') as f:
            f.write(info)
    return model_list, x_train, r_train, y_train, x_test, r_test, y_test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='adult',
                        choices= ['compas', 'loan', 'adult', 'diabetes', 'blog', 'auto'])
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-w', '--write_log', type=int, default=0)
    args, unknown = parser.parse_known_args()
    print(f'dataset: {args.dataset}', flush=True)
    print(f'seed: {args.seed}', flush=True)
    dataset, seed, write_log = args.dataset, args.seed, args.write_log
    model_list, x_train, r_train, y_train, x_test, r_test, y_test = run_exp(dataset, seed, write_log)

