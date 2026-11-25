import os, sys, json, time, random
import numpy as np
import scipy.io as sio
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

OUT_DIR = os.path.expanduser("~/Code/CATS-Net/A04_Word2Vec/reanalysis_r5c2")
DATA_ROOT = os.path.expanduser("~/Data")
WV_PATH = os.path.expanduser("~/Code/CATS-Net/A04_Word2Vec/wordvector/embedding_dim_20.mat")
RN18_PATH = os.path.expanduser("~/Code/CATS-Net/Deps/pretrained_fe/resnet18-f37072fd.pth")
CACHE_PATH = os.path.join(OUT_DIR, "cache_cifar100_rn18_feats.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class Net2MLP(nn.Module):
    def __init__(self, context_dim=20, feature_dim=512):
        super().__init__()
        self.clf_fc1 = nn.Sequential(nn.Linear(feature_dim, 100, bias=False), nn.BatchNorm1d(100), nn.ReLU())
        self.clf_fc2 = nn.Sequential(nn.Linear(100, 10, bias=False), nn.BatchNorm1d(10), nn.ReLU())
        self.clf_fc3 = nn.Sequential(nn.Linear(10, 2))
        self.cdp_fc1 = nn.Sequential(nn.Linear(context_dim, feature_dim, bias=False), nn.BatchNorm1d(feature_dim), nn.Sigmoid())
        self.cdp_fc2 = nn.Sequential(nn.Linear(feature_dim, 100, bias=False), nn.BatchNorm1d(100), nn.Sigmoid())
        self.cdp_fc3 = nn.Sequential(nn.Linear(100, 10, bias=False), nn.BatchNorm1d(10), nn.Sigmoid())

    def forward(self, x, contexts):
        y = self.cdp_fc1(contexts)
        x = self.clf_fc1(x * y)
        y = self.cdp_fc2(y)
        x = self.clf_fc2(x * y)
        y = self.cdp_fc3(y)
        x = self.clf_fc3(x * y)
        return x

def extract_or_load_cifar100(n_aug=5):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached CIFAR-100 features from {CACHE_PATH}", flush=True)
        return torch.load(CACHE_PATH, map_location="cpu")

    print("Extracting frozen ResNet18 features for CIFAR-100 (exact original 32x32 norm)...", flush=True)
    t0 = time.time()
    rn18 = models.resnet18()
    rn18.load_state_dict(torch.load(RN18_PATH, map_location="cpu"))
    rn18.fc = nn.Identity()
    rn18 = rn18.to(DEVICE).eval()

    mean = [n / 255.0 for n in [129.3, 124.1, 112.4]]
    std = [n / 255.0 for n in [68.2, 65.4, 70.4]]
    tf_clean = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    tf_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tr_clean_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=tf_clean)
    te_clean_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=tf_clean)

    def extract_loader(ds):
        dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        feats, labs = [], []
        with torch.no_grad():
            for x, y in dl:
                feats.append(rn18(x.to(DEVICE)).half().cpu())
                labs.append(y)
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    tr_clean, tr_labels = extract_loader(tr_clean_ds)
    te_clean, te_labels = extract_loader(te_clean_ds)

    torch.manual_seed(42)
    aug_views = []
    for v in range(n_aug):
        tr_aug_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=tf_aug)
        f_v, _ = extract_loader(tr_aug_ds)
        aug_views.append(f_v)
    tr_aug = torch.stack(aug_views, dim=0)

    data = {
        "tr_clean": tr_clean, "tr_aug": tr_aug, "tr_labels": tr_labels,
        "te_clean": te_clean, "te_labels": te_labels,
    }
    torch.save(data, CACHE_PATH)
    print(f"Saved CIFAR-100 cached features in {time.time()-t0:.1f}s", flush=True)
    return data

@torch.no_grad()
def eval_val(model, feats, labels, embs_gpu, neg_pool_gpu, n_neg_draws=5, seed=777):
    model.eval()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    n = feats.shape[0]
    pos_acc = (model(feats, embs_gpu[labels]).argmax(dim=1) == 1).float().mean().item() * 100.0
    pool_len = neg_pool_gpu.shape[0]
    neg_accs = []
    for _ in range(n_neg_draws):
        ridx = torch.randint(0, pool_len, (n,), generator=g, device=DEVICE)
        neg_cls = neg_pool_gpu[ridx]
        same = (neg_cls == labels)
        while same.any():
            ridx2 = torch.randint(0, pool_len, (int(same.sum().item()),), generator=g, device=DEVICE)
            neg_cls[same] = neg_pool_gpu[ridx2]
            same = (neg_cls == labels)
        neg_accs.append((model(feats, embs_gpu[neg_cls]).argmax(dim=1) == 0).float().mean().item() * 100.0)
    return 0.5 * (pos_acc + float(np.mean(neg_accs))), pos_acc, float(np.mean(neg_accs))

@torch.no_grad()
def eval_heldout_exact(model, te_feats, test_id, embs_gpu, other_ids_gpu):
    """Exact evaluation on held-out class: all 100 images vs true concept (pos) and vs ALL 99 other concepts (neg)."""
    model.eval()
    n = te_feats.shape[0]
    pos_c = embs_gpu[test_id].unsqueeze(0).expand(n, -1)
    pos_acc = (model(te_feats, pos_c).argmax(dim=1) == 1).float().mean().item() * 100.0

    # Exact expectation over all 99 negative classes (100 * 99 = 9900 pairs)
    n_neg = other_ids_gpu.shape[0]
    feats_rep = te_feats.unsqueeze(1).expand(n, n_neg, -1).reshape(n * n_neg, -1)
    neg_c_rep = embs_gpu[other_ids_gpu].unsqueeze(0).expand(n, n_neg, -1).reshape(n * n_neg, -1)
    neg_acc_exact = (model(feats_rep, neg_c_rep).argmax(dim=1) == 0).float().mean().item() * 100.0

    # Also 1 random draw matching original test()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(999 + test_id)
    ridx = torch.randint(0, n_neg, (n,), generator=g, device=DEVICE)
    neg_acc_1d = (model(te_feats, embs_gpu[other_ids_gpu[ridx]]).argmax(dim=1) == 0).float().mean().item() * 100.0
    return 0.5 * (pos_acc + neg_acc_exact), pos_acc, neg_acc_exact, 0.5 * (pos_acc + neg_acc_1d)

def main():
    data = extract_or_load_cifar100(n_aug=5)
    tr_clean = data["tr_clean"].float().to(DEVICE)
    tr_aug = data["tr_aug"].float().to(DEVICE)
    tr_labels_all = data["tr_labels"].to(DEVICE)
    te_clean = data["te_clean"].float().to(DEVICE)
    te_labels_all = data["te_labels"].to(DEVICE)

    wv = sio.loadmat(WV_PATH)["wv_d_20"].astype(np.float32) * 10.0  # exact original *10 scaling
    embs_gpu = torch.tensor(wv, dtype=torch.float32, device=DEVICE)  # (100, 20)

    tr_cls_idx = [torch.where(tr_labels_all == c)[0] for c in range(100)]
    te_cls_idx = [torch.where(te_labels_all == c)[0] for c in range(100)]

    epochs = 200
    batch_size = 256
    ni = 0.05
    lr = 1e-4
    n_val_classes = 10

    results = []
    t_start = time.time()
    print("Running Word2Vec (CIFAR-100) Scheme A L1O across all 100 classes (89 inner-train / 10 inner-val / 1 held-out)...", flush=True)

    for test_id in range(100):
        seed = 2000 + test_id
        rng = random.Random(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        avail_classes = [c for c in range(100) if c != test_id]
        val_classes = sorted(rng.sample(avail_classes, n_val_classes))
        val_set = set(val_classes)
        tr_classes = [c for c in avail_classes if c not in val_set]

        tr_idx = torch.cat([tr_cls_idx[c] for c in tr_classes])
        val_idx = torch.cat([tr_cls_idx[c] for c in val_classes])
        te_idx = te_cls_idx[test_id]

        val_feats = tr_clean[val_idx]
        val_labels = tr_labels_all[val_idx]
        te_feats = te_clean[te_idx]

        tr_pool_gpu = torch.tensor(tr_classes, dtype=torch.long, device=DEVICE)
        avail_pool_gpu = torch.tensor(avail_classes, dtype=torch.long, device=DEVICE)

        model = Net2MLP(context_dim=20, feature_dim=512).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_epoch = -1
        best_state = None
        n_tr = tr_idx.shape[0]
        n_aug = tr_aug.shape[0]

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_tr, device=DEVICE)
            view_ids = torch.randint(0, n_aug, (n_tr,), device=DEVICE)
            ep_idx = tr_idx[perm]
            ep_x = tr_aug[view_ids, ep_idx]
            ep_y = tr_labels_all[ep_idx]

            for s in range(0, n_tr, batch_size):
                bx = ep_x[s:s+batch_size]
                by = ep_y[s:s+batch_size]
                bsz = bx.shape[0]
                half = bsz // 2
                ridx = torch.randint(0, tr_pool_gpu.shape[0], (bsz - half,), device=DEVICE)
                neg_c_ids = tr_pool_gpu[ridx]
                same = (neg_c_ids == by[half:])
                while same.any():
                    ridx2 = torch.randint(0, tr_pool_gpu.shape[0], (int(same.sum().item()),), device=DEVICE)
                    neg_c_ids[same] = tr_pool_gpu[ridx2]
                    same = (neg_c_ids == by[half:])
                c_ids = torch.cat([by[:half], neg_c_ids], dim=0)
                bin_y = torch.zeros(bsz, dtype=torch.long, device=DEVICE)
                bin_y[:half] = 1
                bc = embs_gpu[c_ids] + torch.empty((bsz, 20), device=DEVICE).uniform_(-ni, ni)

                opt.zero_grad(set_to_none=True)
                out = model(bx, bc)
                loss = crit(out, bin_y)
                loss.backward()
                opt.step()

            if ep % 2 == 0 or ep == epochs - 1:
                val_acc, _, _ = eval_val(model, val_feats, val_labels, embs_gpu, avail_pool_gpu, n_neg_draws=5, seed=777)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Diagnostic only: final epoch held-out acc
        last_acc, _, _, _ = eval_heldout_exact(model, te_feats, test_id, embs_gpu, avail_pool_gpu)

        # SINGLE INDEPENDENT EVALUATION of fixed best_inner_val model on held-out class test_id
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        te_acc, te_pos, te_neg, te_acc_1d = eval_heldout_exact(model, te_feats, test_id, embs_gpu, avail_pool_gpu)

        results.append({
            "class_id": test_id,
            "best_val_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "heldout_acc": te_acc,
            "heldout_pos_acc": te_pos,
            "heldout_neg_acc": te_neg,
            "heldout_acc_1draw": te_acc_1d,
            "last_epoch_heldout_acc": last_acc,
        })
        if (test_id + 1) % 10 == 0 or test_id == 0:
            cur_mean = np.mean([r["heldout_acc"] for r in results])
            cur_val = np.mean([r["best_val_acc"] for r in results])
            print(f"[{test_id+1:3d}/100] cls={test_id:2d} best_ep={best_epoch:3d} val={best_val_acc:.2f}% test={te_acc:.2f}% (pos={te_pos:.1f}%, neg={te_neg:.1f}%) | running_mean={cur_mean:.2f}% (val={cur_val:.2f}%) [{time.time()-t_start:.1f}s]", flush=True)

    accs = np.array([r["heldout_acc"] for r in results])
    accs_1d = np.array([r["heldout_acc_1draw"] for r in results])
    accs_last = np.array([r["last_epoch_heldout_acc"] for r in results])
    val_accs = np.array([r["best_val_acc"] for r in results])
    best_eps = np.array([r["best_val_epoch"] for r in results])

    def summarize(name, arr):
        t_stat, p_val = stats.ttest_1samp(arr, 50.0, alternative="greater")
        d = (arr.mean() - 50.0) / arr.std(ddof=1)
        ci = stats.t.interval(0.95, len(arr)-1, loc=arr.mean(), scale=stats.sem(arr))
        w_stat, w_p = stats.wilcoxon(arr - 50.0, alternative="greater")
        print(f"  {name:28s}: mean={arr.mean():.2f}% ± {arr.std(ddof=1):.2f}%, med={np.median(arr):.2f}%, 95%CI=[{ci[0]:.2f}%, {ci[1]:.2f}%], t={t_stat:.2f}, d={d:.2f}, p={p_val:.2e}, wilcox_p={w_p:.2e}", flush=True)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "median": float(np.median(arr)), "ci95": [float(ci[0]), float(ci[1])], "t": float(t_stat), "d": float(d), "p": float(p_val), "wilcoxon_p": float(w_p)}

    print("\n================ WORD2VEC (CIFAR-100) SCHEME A SUMMARY ================", flush=True)
    s_main = summarize("Held-out (Scheme A, exact)", accs)
    s_1d = summarize("Held-out (Scheme A, 1-draw)", accs_1d)
    s_last = summarize("Held-out (last ep 199)", accs_last)
    print(f"  Inner-val best acc: mean={val_accs.mean():.2f}% ± {val_accs.std(ddof=1):.2f}%", flush=True)
    print(f"  Selected epoch quantiles [0,25,50,75,100]: {np.percentile(best_eps, [0,25,50,75,100])}", flush=True)
    pos_m = np.mean([r["heldout_pos_acc"] for r in results])
    neg_m = np.mean([r["heldout_neg_acc"] for r in results])
    print(f"  Mean pos_acc={pos_m:.2f}%, mean neg_acc={neg_m:.2f}%", flush=True)

    out_json = os.path.join(OUT_DIR, "word2vec_schemeA_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": s_main, "summary_1draw": s_1d, "summary_last_epoch": s_last, "per_class": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_json} (total time {time.time()-t_start:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
