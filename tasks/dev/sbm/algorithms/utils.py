import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
from .register import algorithm_dict
from tqdm import tqdm

def compute_nams(model, images, feature_index, layer_name='layer4'):
    b_size = images.shape[0]
    feature_maps = compute_feature_maps(images, model, layer_name=layer_name)
    nams = (feature_maps[:, feature_index, :, :]).detach()
    nams_flat = nams.view(b_size, -1)
    nams_max, _ = torch.max(nams_flat, dim=1, keepdim=True)
    nams_flat = nams_flat/nams_max
    nams = nams_flat.view_as(nams)

    nams_resized = []
    for nam in nams:
        nam = nam.cpu().numpy()
        nam = cv2.resize(nam, images.shape[2:])
        nams_resized.append(nam)
    nams = np.stack(nams_resized, axis=0)
    nams = torch.from_numpy(1-nams)
    return nams


def compute_heatmaps(imgs, masks):
    heatmaps = []
    for (img, mask) in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = torch.from_numpy(heatmaps).permute(0, 3, 1, 2).clamp(0.0, 1.0)
    return heatmaps


def grad_step(adv_inputs, grad, step_size):
    l = len(adv_inputs.shape) - 1
    grad_norm = torch.norm(
        grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
    scaled_grad = grad / (grad_norm + 1e-10)
    return adv_inputs + scaled_grad * step_size


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax


def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


class RunningVariance:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self):
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)
    
def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def extract_feature_info(
    model,
    loader,
    embedding_dir="test",
    split="train",
    threshold=10000,
    keep_ndims=-1,
    device="cuda:0"
):
    if len(glob.glob(os.path.join(embedding_dir, f"embeds_{split}*"))) > 0:
        print("File exists, skipping")
        embeddings, labels, pred_probs, groups = load_embeddings(
            embedding_dir, split)
        if keep_ndims == -1:
            return embeddings, labels, pred_probs, groups
        else:
            del embeddings
            pca_embeds = disentangle_pca(embedding_dir, split, keep_ndims)
            return pca_embeds, labels, pred_probs, groups

    else:
        os.makedirs(embedding_dir, exist_ok=True)
    num_samples = len(loader.dataset)

    fea_dim = model.fc.weight.data.shape[1]

    if num_samples > threshold:
        embedding_path = os.path.join(embedding_dir, f"embeds_{split}.mmp")
        embeddings = np.memmap(
            embedding_path, dtype="float32", mode="w+", shape=(num_samples, fea_dim)
        )
    else:
        embeddings = np.zeros((num_samples, fea_dim))
    with open(os.path.join(embedding_dir, f"info_{split}.txt"), "w") as f:
        f.write(f"{num_samples}, {fea_dim}")

    model.eval()
    groups = []
    count = 0

    labels, pred_probs = [], []
    with torch.no_grad():
        for data, y, g, a in tqdm(loader, leave=False):
            logits, feas = model(data.to(device), True)
            logits = logits.detach().cpu()
            feas = feas.detach().cpu().numpy()
            probs = F.softmax(logits, dim=1).numpy()

            pred_probs.append(probs)
            embeddings[count: count + len(feas)] = feas.astype(float)
            labels.append(y.numpy())
            groups.append(g.numpy())
            count += len(feas)

    labels = np.concatenate(labels, axis=0)
    pred_probs = np.concatenate(pred_probs)
    groups = np.concatenate(groups)

    if num_samples > threshold:
        embeddings.flush()
    else:
        np.save(os.path.join(embedding_dir, f"embeds_{split}.npy"), embeddings)
    np.save(os.path.join(embedding_dir, f"labels_{split}.npy"), labels)
    np.save(os.path.join(embedding_dir, f"pred_probs_{split}.npy"), pred_probs)
    np.save(os.path.join(embedding_dir, f"groups_{split}.npy"), groups)
  
    return embeddings, labels, pred_probs, groups

def disentangle_pca(embedding_dir : str, split : str, keep_ndims: int):
    from sklearn.decomposition import PCA
    embed_path2 = os.path.join(embedding_dir, f"embeds_{split}.mmp")
    if keep_ndims == -1 or len(glob.glob(os.path.join(embedding_dir, f"embeds_{split}_{keep_ndims}d*"))) > 0:
        embeddings, labels, pred_probs, groups = load_embeddings(
            embedding_dir, split, keep_ndims)
        print("Load existing embeddings")
        return embeddings
    print("Generating PCA embeddings")
    embeddings, labels, pred_probs, groups = load_embeddings(
        embedding_dir, split)
    if split == "train":
        pca_obj = PCA(n_components=keep_ndims)
        new_embeds = pca_obj.fit_transform(embeddings)
        with open(os.path.join(embedding_dir, f"pca_obj_train_{keep_ndims}d.pickle"), "wb") as f:
            pickle.dump(pca_obj, f)
    else:
        pca_path = os.path.join(
            embedding_dir, f"pca_obj_train_{keep_ndims}d.pickle")
        if not os.path.exists(pca_path):
            raise ValueError("No PCA object")
        with open(pca_path, "rb") as f:
            pca_obj = pickle.load(f)
        new_embeds = pca_obj.transform(embeddings)

    explained_ratio = pca_obj.explained_variance_ratio_.sum()*100

    if os.path.exists(embed_path2):
        with open(os.path.join(embedding_dir, f"info_{split}.txt"), "r") as f:
            line = f.readlines()[0]
        shapes = line.split(",")
        num_samples, fea_dim = int(shapes[0].strip()), int(shapes[1].strip())
        embedding_path = os.path.join(
            embedding_dir, f"embeds_{split}_{keep_ndims}d_{explained_ratio:.3f}r.mmp")
        new_embeds_mmp = np.memmap(
            embedding_path, dtype="float32", mode="w+", shape=(num_samples, keep_ndims)
        )
        new_embeds_mmp[:] = new_embeds[:]
        new_embeds_mmp.flush()
    else:
        np.save(os.path.join(embedding_dir,
                f"embeds_{split}_{keep_ndims}d_{explained_ratio:.3f}r.npy"), new_embeds)
    return new_embeds

def load_embeddings(path, split, keep_ndims=-1):
    if keep_ndims == -1:
        embed_path1 = os.path.join(path, f"embeds_{split}.npy")
        embed_path2 = os.path.join(path, f"embeds_{split}.mmp")
        if os.path.exists(embed_path1):
            embeddings = np.load(embed_path1, allow_pickle=True)
        elif os.path.exists(embed_path2):
            with open(os.path.join(path, f"info_{split}.txt"), "r") as f:
                line = f.readlines()[0]
            shapes = line.split(",")
            embed_shape = (int(shapes[0].strip()), int(shapes[1].strip()))
            embeddings = np.memmap(
                embed_path2, dtype="float32", mode="r", shape=embed_shape
            )
        else:
            raise ValueError("Embedding not exists")
    else:
        embed_path1 = os.path.join(path, f"embeds_{split}_{keep_ndims}d*.npy")
        embed_path2 = os.path.join(path, f"embeds_{split}_{keep_ndims}d*.mmp")
        if len(glob.glob(embed_path1)) > 0:
            embed_path = glob.glob(embed_path1)[0]
            print("load", embed_path)
            embeddings = np.load(embed_path, allow_pickle=True)
        elif len(glob.glob(embed_path2)) > 0:
            with open(os.path.join(path, f"info_{split}.txt"), "r") as f:
                line = f.readlines()[0]
            shapes = line.split(",")
            embed_shape = (int(shapes[0].strip()), keep_ndims)
            embed_path = glob.glob(embed_path2)[0]
            print("load", embed_path)
            embeddings = np.memmap(
                embed_path, dtype="float32", mode="r", shape=embed_shape
            )
        else:
            raise ValueError("Embedding not exists")
    labels = np.load(os.path.join(
        path, f"labels_{split}.npy"), allow_pickle=True)
    pred_probs = np.load(os.path.join(
        path, f"pred_probs_{split}.npy"), allow_pickle=True)
    groups = np.load(os.path.join(
        path, f"groups_{split}.npy"), allow_pickle=True)
    return embeddings, labels, pred_probs, groups



def get_misclassified_samples(model, idx_train_loader, device):
    mis_classified = []
    model.eval()
    with torch.no_grad():
        for idx, data, y, g, a in idx_train_loader:
            data = data.to(device)
            logits = model.fc(data)
            preds = torch.argmax(logits, dim=-1).cpu()
            mis_classified.append(idx[preds != y])
    mis_classified = torch.cat(mis_classified)
    return mis_classified

# https://github.com/izmailovpavel/spurious_feature_learning/blob/main/optimizers/__init__.py
def bert_adamw_optimizer(model, lr, weight_decay):
    # Adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def init_optimizer(model, opt_name, kwargs):
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"], weight_decay=kwargs["weight_decay"])
    elif opt_name == "adam":
        optimizer = torch.optim.SGD(model.parameters(), lr=kwargs["lr"])
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
    elif opt_name == "bert_adamw":
        optimizer = bert_adamw_optimizer(model, kwargs["lr"], kwargs["weight_decay"])
    else:
        raise ValueError("Unknown optimizer")
        
    return optimizer

def init_scheduler(optimizer, scheduler_name, kwargs):
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs["T_max"]
        )
    elif scheduler_name == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=kwargs["milestones"], gamma=kwargs["gamma"],
        )
    elif scheduler_name == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=kwargs["total_iters"])
    else:
        scheduler = None
    return scheduler


def init_algorithm(config):
    if config.algorithm not in algorithm_dict:
        raise ValueError("Invalid algorithm name")
    return algorithm_dict[config.algorithm](config)