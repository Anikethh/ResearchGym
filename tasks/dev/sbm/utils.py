import os
import subprocess
import shlex
import numpy as np
import time
import torch
from datetime import datetime
import shutil

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_gpu(gpu):
    print("set gpu:", gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_free_gpu():
    gpu_info = subprocess.Popen(
        shlex.split("nvidia-smi -q -d Memory"), stdout=subprocess.PIPE, text=True
    )
    grep1 = subprocess.Popen(
        shlex.split("grep -A4 GPU"),
        stdin=gpu_info.stdout,
        stdout=subprocess.PIPE,
        text=True,
    )
    grep2 = subprocess.Popen(
        shlex.split("grep Used"), stdin=grep1.stdout, stdout=subprocess.PIPE, text=True
    )
    output, error = grep2.communicate()
    memory_available = np.array(
        [int(x.split(":")[1].strip().split()[0])
         for x in output.split("\n")[0:-1]]
    )
    return np.argsort(memory_available)


def time_str(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    if t >= 60:
        return "{:.1f}m".format(t / 60)
    return "{:.1f}s".format(t)


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def prepare_experiment(args):
    if len(args.save_folder) == 0:
        from config import EXPR_PATH
        save_folder = EXPR_PATH
    else:
        save_folder = args.save_folder
    # prepare the experiment folder
    if args.split_train < 1.0:
        expr_name = f"{args.algorithm}_{args.dataset}_{args.backbone}_{args.train_split}_{args.split_train:.2f}train_{args.batch_size}B_{args.epoch}E_seed{args.seed}{args.tag}"
    else:
        expr_name = f"{args.algorithm}_{args.dataset}_{args.backbone}_{args.train_split}_{args.batch_size}B_{args.epoch}E_seed{args.seed}{args.tag}"
    if args.algorithm == "erm":
        if args.pretrained == True:
            expr_name += "_pretrained"
        else:
            expr_name += "_scratch"
    # now = datetime.now()
    # timestamp = now.strftime("%m%d%Y-%H%M%S")
    # expr_name += f"_{timestamp}"

    if args.mode == "debug":
        save_path = os.path.join(save_folder, f"{args.dataset}_debug")
        os.makedirs(save_path, exist_ok=True)
        set_log_path(save_path)
    elif args.mode == "train":
        save_path = os.path.join(save_folder, expr_name)
        os.makedirs(save_path, exist_ok=True)
        set_log_path(save_path)
    elif args.mode == "test":
        save_path = os.path.join(save_folder, expr_name)
        if os.path.exists(args.check_point):
            set_log_path(args.check_point)
        elif os.path.exists(save_path):
            set_log_path(save_path)
        else:
            raise ValueError(f"Checkpoint file {args.check_point} does not exist.")
    else:
        raise ValueError(f"Unknow running mode {args.mode}. Should be debug, train, or test")
    if args.algorithm == "lasar":
        score_path = os.path.join(
            save_folder, f"{args.dataset}_scores")
        os.makedirs(score_path, exist_ok=True)
    

    args_str = (
        "--------Parameters--------\n"
        + "\n".join(["{}={}".format(k, args.__dict__[k])
                    for k in args.__dict__])
        + "\n--------------------"
    )
    log(args_str)
    set_seed(args.seed)
    if args.mode == "train":
        current_directory = os.path.dirname(os.path.realpath(__file__))
        shutil.copytree(current_directory, os.path.join(save_path,'source'),
                            ignore=shutil.ignore_patterns('*.ipynb', '*.txt','*.png', '*.sh', '*.csv', '*.svg',
                                                        '*.md','*.pickle','*.jpg','.vscode', 'figures',
                                                        'afr','celeba_figures','deep_feature_reweighting','salient_imagenet', 
                                                        '__pycache__','.git', '.ipynb_checkpoints','*.pyc','wilds','SubpopBench', "not_used"
                                ),
                            dirs_exist_ok = True
                        )
    return save_path
    
class BestMetric:
    def __init__(self, max_val=True):
        self.best_val = -float("inf")
        self.max_val = max_val
    def add(self, val):
        if not self.max_val:
            val = -val
        if self.best_val < val:
            self.best_val = val
            return 1
        else:
            return 0

    def get(self):
        if self.max_val:
            return self.best_val
        else:
            return -self.best_val


class BestMetricGroup(BestMetric):
    def __init__(self):
        super(BestMetricGroup, self).__init__()

    def str(self):
        test_avg_acc = self.best_test[0]
        test_unbiased_accs = self.best_test[1]
        test_worst_acc = self.best_test[2]

        val_acc = self.best_val
        res_str = f"val: {val_acc:.6f} test: avg_acc {test_avg_acc:.6f} worst_acc {test_worst_acc:.6f} test_unbiased {test_unbiased_accs:.6f} val_avg {self.best_test[3]}"
        return res_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
