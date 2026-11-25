import os, sys, glob, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy import stats

OUT_DIR = os.path.expanduser("~/Code/CATS-Net/A04_Word2Vec/reanalysis_r5c2")
THINGS_ROOT = os.path.expanduser("~/Data/THINGS")
IMG_ROOT = os.path.join(THINGS_ROOT, "object_images")
EMB_PATH = os.path.join(THINGS_ROOT, "spose_embedding_49d_sorted.txt")
TSV_PATH = os.path.join(THINGS_ROOT, "things_concepts.tsv")
RN18_PATH = os.path.expanduser("~/Code/CATS-Net/Deps/pretrained_fe/resnet18-f37072fd.pth")
ORIG_RESULTS = os.path.expanduser("~/Code/CATS-Net/A04_Word2Vec/THINGS_results")
CACHE_PATH = os.path.join(OUT_DIR, "cache_things_rn18_feats.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class Net2MLP(nn.Module):
    """Exact Net2 architecture from CATSnet.py operating on pre-extracted frozen ResNet18 512-d features."""
    def __init__(self, context_dim=49, feature_dim=512):
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

def load_things_meta():
    with open(TSV_PATH, "r", encoding="utf-8") as f:
        hdr = f.readline()
        uids = [line.strip().split("\t")[1] for line in f]
    embs = np.loadtxt(EMB_PATH, dtype=np.float32)
    assert len(uids) == embs.shape[0] == 1854, f"Mismatch: {len(uids)} vs {embs.shape}"
    return uids, embs

def extract_or_load_features(uids, n_aug=4):
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached THINGS features from {CACHE_PATH}", flush=True)
        return torch.load(CACHE_PATH, map_location="cpu")

    print("Extracting frozen ResNet18 features for all 1854 THINGS classes...", flush=True)
    t0 = time.time()
    rn18 = models.resnet18()
    rn18.load_state_dict(torch.load(RN18_PATH, map_location="cpu"))
    rn18.fc = nn.Identity()
    rn18 = rn18.to(DEVICE).eval()

    tf_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tf_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_paths = []
    labels = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG")
    for cid, uid in enumerate(uids):
        cdir = os.path.join(IMG_ROOT, uid)
        files = sorted([f for f in os.listdir(cdir) if f.endswith(exts)])
        for fn in files:
            img_paths.append(os.path.join(cdir, fn))
            labels.append(cid)
    n_imgs = len(img_paths)
    labels_t = torch.tensor(labels, dtype=torch.long)
    print(f"Total THINGS images: {n_imgs} across {len(uids)} classes", flush=True)

    # Extract clean + n_aug views in batches
    bs = 256
    clean_feats = torch.empty((n_imgs, 512), dtype=torch.float16)
    aug_feats = torch.empty((n_aug, n_imgs, 512), dtype=torch.float16)

    torch.manual_seed(42)
    random.seed(42)
    with torch.no_grad():
        for start in range(0, n_imgs, bs):
            end = min(start + bs, n_imgs)
            pils = [Image.open(p).convert("RGB") for p in img_paths[start:end]]
            x_c = torch.stack([tf_clean(im) for im in pils]).to(DEVICE)
            clean_feats[start:end] = rn18(x_c).half().cpu()
            for v in range(n_aug):
                x_a = torch.stack([tf_aug(im) for im in pils]).to(DEVICE)
                aug_feats[v, start:end] = rn18(x_a).half().cpu()
            if (start // bs) % 25 == 0:
                print(f"  extracted {end}/{n_imgs} ({time.time()-t0:.1f}s)", flush=True)

    data = {"clean": clean_feats, "aug": aug_feats, "labels": labels_t, "uids": uids}
    torch.save(data, CACHE_PATH)
    print(f"Saved cached features in {time.time()-t0:.1f}s", flush=True)
    return data

def get_100_unique_test_ids(uids):
    uid_to_idx = {u: i for i, u in enumerate(uids)}
    orig_names = []
    for p in sorted(glob.glob(os.path.join(ORIG_RESULTS, "*", "experiment_metadata.json"))):
        d = json.load(open(p))
        for nm in d["experiment_config"]["selected_test_class_names"]:
            if nm in uid_to_idx and uid_to_idx[nm] not in orig_names:
                orig_names.append(uid_to_idx[nm])
    rng = random.Random(42)
    remaining = [i for i in range(len(uids)) if i not in set(orig_names)]
    extras = rng.sample(remaining, 100 - len(orig_names))
    selected = orig_names + extras
    assert len(selected) == 100 and len(set(selected)) == 100
    return selected, len(orig_names)

@torch.no_grad()
def eval_on_subset(model, feats, labels, embs_gpu, neg_pool_gpu, n_neg_draws=10, seed=0):
    """Exact balanced (pos + neg)/2 evaluation using true global class IDs."""
    model.eval()
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    n = feats.shape[0]
    # Positive pass (all samples with their TRUE concept vector)
    pos_c = embs_gpu[labels]
    logits_pos = model(feats, pos_c)
    pos_acc = (logits_pos.argmax(dim=1) == 1).float().mean().item() * 100.0

    # Negative pass (averaged over n_neg_draws strictly non-matching concepts from neg_pool_gpu)
    pool_len = neg_pool_gpu.shape[0]
    neg_accs = []
    for _ in range(n_neg_draws):
        ridx = torch.randint(0, pool_len, (n,), generator=g, device=DEVICE)
        neg_cls = neg_pool_gpu[ridx]
        # Ensure neg_cls != labels
        same = (neg_cls == labels)
        while same.any():
            ridx2 = torch.randint(0, pool_len, (int(same.sum().item()),), generator=g, device=DEVICE)
            neg_cls[same] = neg_pool_gpu[ridx2]
            same = (neg_cls == labels)
        neg_c = embs_gpu[neg_cls]
        logits_neg = model(feats, neg_c)
        neg_accs.append((logits_neg.argmax(dim=1) == 0).float().mean().item() * 100.0)
    neg_acc_mean = float(np.mean(neg_accs))
    neg_acc_single = float(neg_accs[0])
    return 0.5 * (pos_acc + neg_acc_mean), pos_acc, neg_acc_mean, 0.5 * (pos_acc + neg_acc_single)

def main():
    uids, embs_np = load_things_meta()
    feat_data = extract_or_load_features(uids, n_aug=4)
    clean_all = feat_data["clean"].float().to(DEVICE)   # (26107, 512)
    aug_all = feat_data["aug"].float().to(DEVICE)       # (4, 26107, 512)
    labels_all = feat_data["labels"].to(DEVICE)         # (26107,)
    embs_gpu = torch.tensor(embs_np, dtype=torch.float32, device=DEVICE)  # (1854, 49)

    selected_test_ids, n_orig = get_100_unique_test_ids(uids)
    print(f"Running SPOSE49 Scheme A L1O on 100 unique classes ({n_orig} from original + {100-n_orig} deterministic extras)", flush=True)

    # Pre-build index lists per class on GPU for instant slicing
    cls_indices = [torch.where(labels_all == c)[0] for c in range(1854)]

    epochs = 100
    batch_size = 1024
    ni = 0.1
    lr = 1e-4
    n_val_classes = 100

    results = []
    t_start = time.time()

    for run_idx, test_id in enumerate(selected_test_ids):
        seed = 1000 + test_id
        rng = random.Random(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Strictly isolate test_id; split remaining 1853 classes into 1753 inner-train + 100 inner-val
        avail_classes = [c for c in range(1854) if c != test_id]
        val_classes = sorted(rng.sample(avail_classes, n_val_classes))
        val_set = set(val_classes)
        tr_classes = [c for c in avail_classes if c not in val_set]

        tr_idx = torch.cat([cls_indices[c] for c in tr_classes])
        val_idx = torch.cat([cls_indices[c] for c in val_classes])
        te_idx = cls_indices[test_id]

        tr_labels = labels_all[tr_idx]
        val_feats = clean_all[val_idx]
        val_labels = labels_all[val_idx]
        te_feats = clean_all[te_idx]
        te_labels = labels_all[te_idx]

        tr_pool_gpu = torch.tensor(tr_classes, dtype=torch.long, device=DEVICE)
        avail_pool_gpu = torch.tensor(avail_classes, dtype=torch.long, device=DEVICE)

        model = Net2MLP(context_dim=49, feature_dim=512).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_epoch = -1
        best_state = None
        n_tr = tr_idx.shape[0]
        n_aug = aug_all.shape[0]

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n_tr, device=DEVICE)
            view_ids = torch.randint(0, n_aug, (n_tr,), device=DEVICE)
            ep_idx = tr_idx[perm]
            ep_x = aug_all[view_ids, ep_idx]
            ep_y = labels_all[ep_idx]

            for s in range(0, n_tr, batch_size):
                bx = ep_x[s:s+batch_size]
                by = ep_y[s:s+batch_size]
                bsz = bx.shape[0]
                half = bsz // 2
                # Strictly 50/50 pos/neg within inner-train classes
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
                bc = embs_gpu[c_ids] + torch.empty((bsz, 49), device=DEVICE).uniform_(-ni, ni)

                opt.zero_grad(set_to_none=True)
                out = model(bx, bc)
                loss = crit(out, bin_y)
                loss.backward()
                opt.step()

            # Inner validation ONLY on unseen inner-val classes (NEVER touches held-out test_id!)
            val_acc, _, _, _ = eval_on_subset(model, val_feats, val_labels, embs_gpu, avail_pool_gpu, n_neg_draws=5, seed=777)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Diagnostic only: final epoch held-out acc
        last_acc, last_pos, last_neg, _ = eval_on_subset(model, te_feats, te_labels, embs_gpu, avail_pool_gpu, n_neg_draws=50, seed=999)

        # SINGLE INDEPENDENT EVALUATION of fixed best_inner_val model on held-out class test_id
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        te_acc, te_pos, te_neg, te_acc_1draw = eval_on_subset(model, te_feats, te_labels, embs_gpu, avail_pool_gpu, n_neg_draws=50, seed=999)

        results.append({
            "run_idx": run_idx,
            "class_id": test_id,
            "class_name": uids[test_id],
            "from_orig_95": run_idx < n_orig,
            "n_test_imgs": int(te_idx.shape[0]),
            "best_val_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "heldout_acc": te_acc,
            "heldout_pos_acc": te_pos,
            "heldout_neg_acc": te_neg,
            "heldout_acc_1draw": te_acc_1draw,
            "last_epoch_heldout_acc": last_acc,
        })
        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            cur_mean = np.mean([r["heldout_acc"] for r in results])
            cur_val = np.mean([r["best_val_acc"] for r in results])
            print(f"[{run_idx+1:3d}/100] cls={test_id:4d} ({uids[test_id]:16s}) best_ep={best_epoch:2d} val={best_val_acc:.2f}% test={te_acc:.2f}% (pos={te_pos:.1f}%, neg={te_neg:.1f}%) | running_mean={cur_mean:.2f}% (val={cur_val:.2f}%) [{time.time()-t_start:.1f}s]", flush=True)

    # Summary statistics
    accs = np.array([r["heldout_acc"] for r in results])
    accs_1d = np.array([r["heldout_acc_1draw"] for r in results])
    accs_last = np.array([r["last_epoch_heldout_acc"] for r in results])
    accs_orig95 = np.array([r["heldout_acc"] for r in results if r["from_orig_95"]])
    val_accs = np.array([r["best_val_acc"] for r in results])
    best_eps = np.array([r["best_val_epoch"] for r in results])

    def summarize(name, arr):
        t_stat, p_val = stats.ttest_1samp(arr, 50.0, alternative="greater")
        d = (arr.mean() - 50.0) / arr.std(ddof=1)
        ci = stats.t.interval(0.95, len(arr)-1, loc=arr.mean(), scale=stats.sem(arr))
        w_stat, w_p = stats.wilcoxon(arr - 50.0, alternative="greater")
        print(f"  {name:28s}: mean={arr.mean():.2f}% ± {arr.std(ddof=1):.2f}%, med={np.median(arr):.2f}%, 95%CI=[{ci[0]:.2f}%, {ci[1]:.2f}%], t={t_stat:.2f}, d={d:.2f}, p={p_val:.2e}, wilcox_p={w_p:.2e}", flush=True)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "median": float(np.median(arr)), "ci95": [float(ci[0]), float(ci[1])], "t": float(t_stat), "d": float(d), "p": float(p_val), "wilcoxon_p": float(w_p)}

    print("\n================ SPOSE49 (THINGS) SCHEME A SUMMARY ================", flush=True)
    s_main = summarize("Held-out (Scheme A, 50-draw)", accs)
    s_1d = summarize("Held-out (Scheme A, 1-draw)", accs_1d)
    s_orig = summarize("Held-out (orig 95 classes)", accs_orig95)
    s_last = summarize("Held-out (last ep 99)", accs_last)
    print(f"  Inner-val best acc: mean={val_accs.mean():.2f}% ± {val_accs.std(ddof=1):.2f}%", flush=True)
    print(f"  Selected epoch quantiles [0,25,50,75,100]: {np.percentile(best_eps, [0,25,50,75,100])}", flush=True)
    print(f"  Mean pos_acc={np.mean([r['heldout_pos_acc'] for r in results]):.2f}%, mean neg_acc={np.mean([r['heldout_neg_acc'] for r in results]):.2f}%", flush=True)

    out_json = os.path.join(OUT_DIR, "spose49_schemeA_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": s_main, "summary_1draw": s_1d, "summary_orig95": s_orig, "summary_last_epoch": s_last, "per_class": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_json} (total time {time.time()-t_start:.1f}s)", flush=True)

if __name__ == "__main__":
    main()
