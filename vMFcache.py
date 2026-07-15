"""vMFcache: cache-based DQDA + vMF mixture TTA.

Migrated from ADAPT_online_bayes_cache_dqda.py. DQDA and var_aligned kappa are updated from
cache admissions only (not all streaming batch samples).
"""
import argparse
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ive
from utils.tools import Summary, AverageMeter, accuracy, set_random_seed
from utils.cache_tsne_viz import assign_cache_sid, render_cache_tsne
from data.cls_to_names import custom_scale
from clip import clip
from data.datautils import build_test_loader
import wandb
from datetime import datetime
from tqdm import tqdm

WANDB_PROJECT = "vMFcache"


@torch.no_grad()
def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)


class EmpiricalDMTracker:
    """Online empirical quantiles of D_M(z, pred) for safe-annulus and gamma scaling."""

    def __init__(self, q_low, q_high, min_samples, max_buffer=10000):
        self.q_low = q_low
        self.q_high = q_high
        self.min_samples = min_samples
        self.max_buffer = max_buffer
        self.values = []

    def update(self, dm_tensor):
        vals = dm_tensor.detach().float().cpu().tolist()
        self.values.extend(vals)
        if len(self.values) > self.max_buffer:
            self.values = self.values[-self.max_buffer:]

    def thresholds(self):
        if len(self.values) < self.min_samples:
            return None, None
        t = torch.tensor(self.values)
        return (
            torch.quantile(t, self.q_low).item(),
            torch.quantile(t, self.q_high).item(),
        )

    def all_values_tensor(self):
        return torch.tensor(self.values) if self.values else torch.empty(0)


def max_redundancy_from_sim(sim):
    """sim: [m, m] unit-vector cosine matrix. Return max off-diagonal sim per row."""
    offdiag = sim.clone()
    offdiag.fill_diagonal_(float('-inf'))
    return offdiag.max(dim=1).values


@torch.no_grad()
def class_dispersion_from_cache(vecs, labels, cache_pro, cls_num):
    """Per-class mean resultant R_c from cache members, soft-weighted by cache_pro[:, label] (DQDA-style)."""
    K = cls_num
    dim = vecs.shape[1]
    device = vecs.device
    sum_vec = torch.zeros(K, dim, device=device, dtype=vecs.dtype)
    weight = torch.zeros(K, device=device, dtype=cache_pro.dtype)
    if vecs.numel() == 0:
        return torch.zeros(K, device=device, dtype=vecs.dtype), weight

    slot_w = cache_pro[torch.arange(labels.numel(), device=device), labels]
    sum_vec.index_add_(0, labels, slot_w.unsqueeze(1) * vecs)
    weight.index_add_(0, labels, slot_w)
    w = weight.clamp(min=1e-12)
    R = sum_vec.norm(dim=1) / w
    R = torch.where(weight > 0, R, torch.zeros_like(R))
    return R, weight


def _A_d(kappa, dim):
    """Ratio I_{d/2}(kappa)/I_{d/2-1}(kappa) for vMF mean resultant."""
    nu = dim / 2.0 - 1.0
    kappa = np.asarray(kappa, dtype=np.float64)
    return ive(nu + 1.0, kappa) / (ive(nu, kappa) + 1e-300)


def estimate_kappa_from_R(R, dim, n_iter=64, hi_max=1e8):
    """MLE concentration kappa from mean resultant length R via bisection on A_d(kappa)=R."""
    R = float(np.clip(R, 1e-8, 1.0 - 1e-8))
    hi = 1.0
    while _A_d(hi, dim) < R and hi < hi_max:
        hi *= 2.0
    lo = 0.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if _A_d(mid, dim) < R:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def class_kappa_from_dispersion(R, weight, dim, min_weight, kappa_fallback=None):
    """Per-class MLE kappa from soft-label dispersion; invalid classes use kappa_fallback."""
    K = R.shape[0]
    kappa = torch.full((K,), float('nan'), device=R.device, dtype=torch.float64)
    valid = weight >= min_weight
    for c in torch.where(valid)[0]:
        kappa[c] = estimate_kappa_from_R(R[c].item(), dim)
    if kappa_fallback is not None:
        fb = float(kappa_fallback)
        kappa = torch.where(valid, kappa, torch.full_like(kappa, fb))
    return kappa, valid


def _safe_softmax_rows(logits, dim=1):
    """Softmax with finite rows; all-nonfinite rows become zeros."""
    has_finite = torch.isfinite(logits).any(dim=dim, keepdim=True)
    safe = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e4))
    probs = torch.softmax(safe, dim=dim)
    return torch.where(has_finite, probs, torch.zeros_like(probs))


def log_vmf_normalizer(kappa, dim):
    """log C_d(kappa) for vMF on unit (d-1)-sphere embedded in R^d."""
    nu = dim / 2.0 - 1.0
    kappa = np.clip(np.asarray(kappa, dtype=np.float64), 1e-12, None)
    log_iv = np.log(ive(nu, kappa) + 1e-300) + kappa
    return (dim / 2.0 - 1.0) * np.log(kappa) - (dim / 2.0) * np.log(2 * np.pi) - log_iv


def _report_stats(name, tensor):
    """Print mean/std/min/max/percentiles of a 1D tensor; return (mean, std)."""
    t = tensor.float()
    if t.numel() == 0:
        print(f"[{name}] empty")
        return 0.0, 0.0
    q = torch.quantile(t, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    print(f"[{name}] n={t.numel()} mean={t.mean():.4f} std={t.std():.4f} "
          f"min={t.min():.4f} max={t.max():.4f} "
          f"p5={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p95={q[4]:.4f}")
    return t.mean().item(), t.std().item()


@torch.no_grad()
def compute_mahalanobis(features, mus, var_diag):
    """Squared diagonal Mahalanobis distance. features: [N, D], mus/var_diag: [K, D] -> [N, K]."""
    diff = features.unsqueeze(1) - mus.unsqueeze(0)
    return (diff ** 2 / var_diag.unsqueeze(0)).sum(2)


@torch.no_grad()
def compute_dqda_logits(features, mus, var_diag):
    """Diagonal QDA discriminant scores (log Gaussian density up to a constant)."""
    maha = compute_mahalanobis(features, mus, var_diag)
    log_det = torch.log(var_diag).sum(1)
    return -0.5 * (log_det.unsqueeze(0) + maha)


@torch.no_grad()
def compute_vmf_logprob(features, vecs, labels, cls_num, class_kappa=None, kappa_default=None):
    """vMF log-density: log C_d(kappa_c) + logsumexp(kappa_c z.m) - log|M_c|."""
    N = features.shape[0]
    dim = features.shape[1]
    sim = features @ vecs.T
    if class_kappa is not None:
        kappa_proto = class_kappa[labels]
        sim = sim * kappa_proto.unsqueeze(0)
        nan_cols = torch.isnan(kappa_proto)
        if nan_cols.any():
            sim[:, nan_cols] = float('-inf')
        log_norm_cache = {}
    elif kappa_default is not None:
        sim = sim * kappa_default
        log_norm_scalar = float(log_vmf_normalizer(kappa_default, dim))
    else:
        raise ValueError('compute_vmf_logprob requires class_kappa or kappa_default')
    logp = torch.full((N, cls_num), float('-inf'), device=features.device)
    for c in torch.unique(labels):
        c_int = int(c.item())
        if class_kappa is not None:
            kc = class_kappa[c_int].item()
            if np.isnan(kc):
                continue
            if c_int not in log_norm_cache:
                log_norm_cache[c_int] = float(log_vmf_normalizer(kc, dim))
            log_norm = log_norm_cache[c_int]
        else:
            log_norm = log_norm_scalar
        mask = labels == c
        cnt = mask.sum()
        lse = torch.logsumexp(sim[:, mask], dim=1)
        logp[:, c_int] = lse - torch.log(cnt.float()) + log_norm
    return logp


@torch.no_grad()
def compute_dm_to_pred(features, mus, var_diag, pred_labels):
    """Mahalanobis distance D_M(z, pred) per sample. features [N,D], pred_labels [N]."""
    mu = mus[pred_labels]
    var = var_diag[pred_labels]
    return (((features - mu) ** 2) / var).sum(1).clamp(min=0).sqrt()


@torch.no_grad()
def compute_mixture_posterior(features, mus, var_diag, vecs, labels, prior, cls_num, args,
                              delta_high, class_kappa=None, vmf_disp_diag=None):
    """Bayesian mixture of DQDA (p_L) and vMF memory (p_S), gated by Mahalanobis-based gamma(z).

    Returns log_mix (log normalized mixture, max-subtracted per row), post, and diagnostic dict.
    """
    maha = compute_mahalanobis(features, mus, var_diag)
    log_det = torch.log(var_diag).sum(1)
    logpL = -0.5 * (log_det.unsqueeze(0) + maha)
    P_L = torch.softmax(logpL, dim=1)

    if getattr(args, 'var_aligned_kappa', False) and class_kappa is not None:
        logpS = compute_vmf_logprob(
            features, vecs, labels, cls_num, class_kappa=class_kappa)
        P_S = _safe_softmax_rows(logpS / args.ps_temperature, dim=1)
    else:
        class_kappa = None
        kappa_default = 1.0 / args.kappa_tau
        logpS = compute_vmf_logprob(
            features, vecs, labels, cls_num, kappa_default=kappa_default)
        P_S = _safe_softmax_rows(logpS, dim=1)

    c_hat = logpL.argmax(1)
    D_M = maha.gather(1, c_hat.unsqueeze(1)).squeeze(1).clamp(min=0).sqrt()
    if delta_high is not None and delta_high > 1e-8:
        gamma = (1 - args.eta * (D_M / delta_high) ** args.rho).clamp(0.0, 1.0).unsqueeze(1)
    else:
        gamma = torch.ones(features.shape[0], 1, device=features.device)

    mix = gamma * P_L + (1 - gamma) * P_S
    post = prior.unsqueeze(0) * mix
    post_sum = post.sum(1, keepdim=True)
    post = torch.where(
        torch.isfinite(post_sum) & (post_sum > 1e-12),
        post / post_sum.clamp(min=1e-12),
        torch.full_like(post, 1.0 / cls_num))
    log_mix = torch.log(post.clamp(min=1e-12))
    log_mix = log_mix - log_mix.max(dim=1, keepdim=True)[0]

    mix_diag = {
        'gamma': gamma.squeeze(1).detach().cpu(),
        'P_L_max': P_L.max(1).values.detach().cpu(),
        'P_S_max': P_S.max(1).values.detach().cpu(),
        'post_max': post.max(1).values.detach().cpu(),
        'D_M': D_M.detach().cpu(),
    }
    if getattr(args, 'var_aligned_kappa', False) and class_kappa is not None:
        mix_diag['kappa_c'] = class_kappa.detach().cpu().float()
        dim = features.shape[1]
        kappa_np = class_kappa.detach().cpu().numpy()
        valid_k = ~np.isnan(kappa_np)
        if valid_k.any():
            log_C = torch.full((cls_num,), float('nan'))
            log_C[torch.tensor(np.where(valid_k)[0], dtype=torch.long)] = torch.tensor(
                log_vmf_normalizer(kappa_np[valid_k], dim), dtype=torch.float32)
            mix_diag['log_C_c'] = log_C
        if vmf_disp_diag is not None:
            mix_diag['R_c'] = vmf_disp_diag['R_c']
            mix_diag['weight_c'] = vmf_disp_diag['weight_c']
            mix_diag['kappa_raw_c'] = vmf_disp_diag['kappa_raw_c']
    return log_mix, post, mix_diag


@torch.no_grad()
def fuse_with_clip(clip_logits, log_mix, scale):
    """ADAPT-style multiplicative fusion: clip_logits * exp(log_mix / scale)."""
    return clip_logits * torch.exp(log_mix / scale)


@torch.no_grad()
def init_empty_banks(cls_num, bank_size, dim, device):
    """Production bank init: all slots empty (label=-1) until admission. No text-prototype warm-start."""
    cache_vecs = torch.zeros((cls_num * bank_size, dim), device=device)
    cache_labels = torch.full((cls_num * bank_size,), -1, dtype=torch.long, device=device)
    cache_pro = torch.zeros((cls_num * bank_size, cls_num), device=device)
    cache_loss = torch.full((cls_num * bank_size,), float('inf'), device=device)
    return [cache_vecs, cache_labels, cache_pro, cache_loss]


@torch.no_grad()
def param_estimation(added_sample, banks, initial_mean, prev_mus, alpha, ridge_eps=0.1):
    """Online DQDA (Diagonal QDA) from cache samples on each admission."""
    image_features, pred, img_pro = added_sample
    vecs, labels, cache_pro = banks
    cache_keys = torch.unique(labels)

    mus = prev_mus.clone()
    mask = labels == pred
    selected_vecs = vecs[mask]
    selected_cache_pro = cache_pro[mask, pred].unsqueeze(1)

    new_mu = ((selected_cache_pro * selected_vecs).sum(dim=0) + img_pro[0][pred] * image_features[0]) / (
        selected_cache_pro.sum() + img_pro[0][pred]).unsqueeze(0)
    new_mu = alpha * new_mu + (1 - alpha) * initial_mean[pred]
    mus[pred] = new_mu

    center_vecs = torch.cat([vecs[labels == i] - mus[i].unsqueeze(0) for i in cache_keys])
    K, d = mus.shape
    if center_vecs.shape[0] >= 2:
        pooled_var = center_vecs.var(dim=0, unbiased=True)
    else:
        pooled_var = torch.ones(d, device=mus.device)

    var_diag = pooled_var.unsqueeze(0).expand(K, -1).clone()
    for k in cache_keys:
        class_vecs = vecs[labels == k] - mus[k].unsqueeze(0)
        if class_vecs.shape[0] >= 2:
            class_var = class_vecs.var(dim=0, unbiased=True)
        else:
            class_var = pooled_var
        var_diag[k] = (1 - ridge_eps) * class_var + ridge_eps * pooled_var

    var_diag = var_diag.clamp(min=1e-4)
    return mus, var_diag


def update_knowledge_banks(banks, sample, bank_size, mus, var_diag,
                           delta_low, delta_high, lambda_div,
                           no_delta_high_gate=False, admit_stats=None, diag=None):
    """Admission: Mahalanobis safe-annulus hard gate + diversity-aware union scoring.

    When full: score the union (existing L + candidate) by entropy and redundancy;
    evict the lowest-scoring member (candidate rejected if it scores lowest).
    """
    pred, feature, e, prob_map = sample
    cache_vecs, cache_labels, cache_pro, cache_loss = banks

    # 1. Mahalanobis safe-annulus hard gate (skipped during bootstrap before DQDA params exist)
    dm = None
    if var_diag is not None and delta_low is not None:
        dm = (((feature[0] - mus[pred]) ** 2) / var_diag[pred]).sum().sqrt()
        reject_low = dm < delta_low
        reject_high = (not no_delta_high_gate) and delta_high is not None and dm > delta_high
        if reject_low or reject_high:
            if diag is not None:
                diag['gate_reject'] = diag.get('gate_reject', 0) + 1
            if admit_stats is not None:
                admit_stats['gate_reject'] += 1
                if reject_high:
                    admit_stats['gate_reject_high'] += 1
            return False, banks, None

    if admit_stats is not None and dm is not None and delta_high is not None and dm > delta_high:
        admit_stats['above_delta_high_seen'] += 1

    start_idx = pred * bank_size
    end_idx = start_idx + bank_size
    existing_count = int((cache_labels[start_idx:end_idx] != -1).sum().item())

    # diagnostics: candidate max redundancy vs existing same-class members
    if diag is not None and existing_count >= 1:
        existing_vecs = cache_vecs[start_idx:start_idx + existing_count]
        cand_max_red = (feature[0] @ existing_vecs.T).max()
        diag['redundancy_cand'].append(cand_max_red.detach().cpu().view(1))
        diag['entropy_cand'].append(e.detach().cpu().view(1))

    # 2a. not full: append
    if existing_count < bank_size:
        insert_idx = start_idx + existing_count
        cache_vecs[insert_idx] = feature[0]
        cache_labels[insert_idx] = pred
        cache_pro[insert_idx] = prob_map[0]
        cache_loss[insert_idx] = e
        if admit_stats is not None and dm is not None and delta_high is not None and dm > delta_high:
            admit_stats['above_delta_high_admitted'] += 1
        return True, [cache_vecs, cache_labels, cache_pro, cache_loss], [feature, pred, prob_map]

    # 2b. full: diversity-aware scored replacement over the union (existing L + candidate)
    slot_idxs = torch.arange(start_idx, end_idx, device=cache_vecs.device)
    union_vecs = torch.cat([cache_vecs[slot_idxs], feature], dim=0)   # [L+1, D]
    union_ent = torch.cat([cache_loss[slot_idxs], e.view(1)], dim=0)  # [L+1]
    sim = union_vecs @ union_vecs.T                              # [L+1, L+1]
    m = union_vecs.shape[0]
    max_red = max_redundancy_from_sim(sim)
    score = -union_ent - lambda_div * max_red
    if diag is not None:
        diag['entropy_union'].append(union_ent.detach().cpu())
        diag['redundancy_union'].append(max_red.detach().cpu())
    evict_local = int(score.argmin().item())
    if evict_local == bank_size:
        return False, [cache_vecs, cache_labels, cache_pro, cache_loss], None

    insert_idx = start_idx + evict_local
    cache_vecs[insert_idx] = feature[0]
    cache_labels[insert_idx] = pred
    cache_pro[insert_idx] = prob_map[0]
    cache_loss[insert_idx] = e
    if admit_stats is not None and dm is not None and delta_high is not None and dm > delta_high:
        admit_stats['above_delta_high_admitted'] += 1
    return True, [cache_vecs, cache_labels, cache_pro, cache_loss], [feature, pred, prob_map]


@torch.no_grad()
def encode_batch(images, clip_weights, encoder):
    if isinstance(images, list):
        images = torch.cat(images, dim=0).cuda()
    else:
        images = images.cuda()

    image_features = encoder(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    clip_logits = 100. * image_features.float() @ clip_weights.float()
    loss = calculate_batch_entropy(clip_logits)
    prob_map = clip_logits.softmax(dim=1)
    pred = clip_logits.argmax(dim=1)
    return image_features.float(), clip_logits, loss, prob_map, pred


@torch.no_grad()
def evaluation(val_loader, clip_weights, image_encoder, args, dataset_name):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    initial_mean = clip_weights.T.float()
    mean = clip_weights.T.float()
    accuracies = []

    cls_num, dim = clip_weights.shape[1], clip_weights.shape[0]
    dm_tracker = EmpiricalDMTracker(
        args.chi2_low, args.chi2_high, args.annulus_min_samples)
    print(f"[annulus] empirical: q_low={args.chi2_low}, q_high={args.chi2_high}, "
          f"min_samples={args.annulus_min_samples}")

    prior = torch.full((cls_num,), 1.0 / cls_num, device=mean.device)

    cache = init_empty_banks(cls_num, args.bank_size, dim, mean.device)
    var_diag = None
    print('[dqda] cache-based param_estimation on each admission (alpha shrinkage to text prototype)')
    print('[vmf] cache-based kappa from admitted cache soft labels (DQDA-style)')
    print('[bank] empty init (no text-prototype warm-start)')
    diag = None
    if getattr(args, 'diag_stats', False):
        diag = {
            'entropy_all': [],             'entropy_union': [], 'redundancy_union': [],
            'entropy_cand': [], 'redundancy_cand': [],
            'gamma': [], 'P_L_max': [], 'P_S_max': [], 'post_max': [], 'D_M_batch': [],
            'gate_reject': 0, 'admit_count': 0,
        }
    mix_diag_batches = [] if getattr(args, 'diag_stats', False) else None
    admit_stats = {
        'admitted': 0, 'gate_reject': 0, 'gate_reject_high': 0,
        'above_delta_high_seen': 0, 'above_delta_high_admitted': 0,
    }
    cache_tsne_viz = getattr(args, 'cache_tsne_viz', False)
    cache_sid = None
    tsne_feats = []
    tsne_labels = []
    global_idx = 0
    if cache_tsne_viz:
        cache_sid = torch.full((cls_num * args.bank_size,), -1, dtype=torch.long, device=mean.device)
        print('[cache_tsne] collecting features during evaluation (viz after pass)')

    start_time = time.time()
    for images, targets in tqdm(val_loader, desc='Processed test images: '):
        features, clip_logits, losses, prob_maps, preds = encode_batch(
            images, clip_weights, image_encoder)
        targets = targets.cuda()
        B = targets.size(0)

        if diag is not None:
            diag['entropy_all'].append(losses.detach().cpu())

        if var_diag is not None and dm_tracker is not None:
            dm_batch = compute_dm_to_pred(features, mean, var_diag, preds)
            dm_tracker.update(dm_batch)
            delta_low, delta_high = dm_tracker.thresholds()
        else:
            delta_low, delta_high = None, None

        for j in range(B):
            pred_j = int(preds[j])
            before_slot = None
            existing_before = 0
            if cache_tsne_viz:
                tsne_feats.append(features[j].detach().cpu())
                tsne_labels.append(int(targets[j].item()))
                start_slot = pred_j * args.bank_size
                existing_before = int((cache[1][start_slot:start_slot + args.bank_size] != -1).sum().item())
                if existing_before >= args.bank_size:
                    before_slot = cache[0][start_slot:start_slot + args.bank_size].clone()

            update_sign, cache, added_sample = update_knowledge_banks(
                cache, [pred_j, features[j:j + 1], losses[j], prob_maps[j:j + 1]],
                args.bank_size, mean, var_diag, delta_low, delta_high, args.lambda_div,
                no_delta_high_gate=args.no_delta_high_gate,
                admit_stats=admit_stats, diag=diag)
            if cache_tsne_viz:
                if update_sign:
                    assign_cache_sid(cache, cache_sid, pred_j, args.bank_size, global_idx,
                                     before_slot, existing_before)
                global_idx += 1
            if update_sign:
                admit_stats['admitted'] += 1
                if diag is not None:
                    diag['admit_count'] += 1
                valid_mask = cache[1] != -1
                banks = [t[valid_mask] for t in cache[:3]]
                mean, var_diag = param_estimation(
                    added_sample, banks, initial_mean, prev_mus=mean, alpha=args.alpha)

        if var_diag is not None:
            valid_mask = cache[1] != -1
            bank_vecs = cache[0][valid_mask]
            bank_labels = cache[1][valid_mask]
            bank_pro = cache[2][valid_mask]
            gamma_delta_high = delta_high
            class_kappa = None
            vmf_disp_diag = None
            if getattr(args, 'var_aligned_kappa', False):
                R, weight = class_dispersion_from_cache(
                    bank_vecs, bank_labels, bank_pro, cls_num)
                kappa_fb = getattr(args, 'kappa_fallback', 2000.0)
                class_kappa, _ = class_kappa_from_dispersion(
                    R, weight, features.shape[1], args.kappa_disp_min_weight,
                    kappa_fallback=kappa_fb)
                class_kappa = class_kappa.float()
                vmf_disp_diag = {
                    'R_c': R.detach().cpu(),
                    'weight_c': weight.detach().cpu(),
                    'kappa_raw_c': class_kappa.detach().cpu(),
                }
            log_mix, post, mix_diag = compute_mixture_posterior(
                features, mean, var_diag, bank_vecs, bank_labels, prior, cls_num, args,
                gamma_delta_high, class_kappa=class_kappa, vmf_disp_diag=vmf_disp_diag)
            test_logits = fuse_with_clip(clip_logits, log_mix, args.scale)
            if mix_diag_batches is not None:
                mix_diag_batches.append(mix_diag)
        else:
            test_logits = clip_logits

        acc = accuracy(test_logits, targets, topk=(1,))
        top1.update(acc[0], B)
        _, pred_top = test_logits.topk(1, 1, True, True)
        correct = pred_top.eq(targets.view(-1, 1).expand_as(pred_top))
        for j in range(B):
            accuracies.append(correct[j, 0].float().item() * 100.0)
        wandb.log({"Averaged test accuracy": round(sum(accuracies) / len(accuracies), 2)}, commit=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    wandb.log({"Elapsed time": f"{elapsed_time:.2f} seconds"}, commit=True)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    gate_mode = "no_delta_high_gate" if args.no_delta_high_gate else "default_annulus"
    print(f"[admission summary] mode={gate_mode} dataset={dataset_name} "
          f"admitted={admit_stats['admitted']} gate_rejected={admit_stats['gate_reject']} "
          f"gate_rejected_high={admit_stats['gate_reject_high']} "
          f"above_delta_high_seen={admit_stats['above_delta_high_seen']} "
          f"above_delta_high_admitted={admit_stats['above_delta_high_admitted']}")

    if diag is not None:
        print(f"\n=== Admission diagnostics ({dataset_name}) ===")
        ent_all = torch.cat(diag['entropy_all']) if diag['entropy_all'] else torch.empty(0)
        _report_stats('clip_entropy (all samples)', ent_all)
        print(f"[admission] admitted={diag['admit_count']} gate_rejected={diag.get('gate_reject', 0)}")

        if dm_tracker is not None:
            dm_all = dm_tracker.all_values_tensor()
            if dm_all.numel() > 0:
                _report_stats('D_M (empirical buffer)', dm_all)
                dl, dh = dm_tracker.thresholds()
                if dl is not None:
                    print(f"[annulus] empirical thresholds: delta_low={dl:.4f}, delta_high={dh:.4f}")

        if diag['redundancy_cand']:
            ec = torch.cat(diag['entropy_cand'])
            rc = torch.cat(diag['redundancy_cand'])
            _, ec_std = _report_stats('entropy (candidates, >=1 same-class member)', ec)
            _, rc_std = _report_stats('max_redundancy (candidate vs existing same-class)', rc)
            print(f"[hint] #admission decisions with existing members={rc.numel()}")
            if ec_std > 0 and rc_std > 0:
                print(f"[hint] lambda_div * std(redundancy) / std(entropy) = "
                      f"{args.lambda_div * rc_std / ec_std:.4f} (should be << 1 for entropy dominance)")
        else:
            print("[diag] no admission decision had an existing same-class member yet.")

        if diag['entropy_union']:
            eu = torch.cat(diag['entropy_union'])
            ru = torch.cat(diag['redundancy_union'])
            _, eu_std = _report_stats('entropy (union @ full-bank decisions)', eu)
            _, ru_std = _report_stats('max_redundancy (union @ full-bank decisions)', ru)
            print(f"[hint] #full-bank comparisons={len(diag['entropy_union'])}")
            if eu_std > 0 and ru_std > 0:
                print(f"[hint] lambda_div * std(redundancy) / std(entropy) @ full-bank = "
                      f"{args.lambda_div * ru_std / eu_std:.4f} (should be << 1 for entropy dominance)")
        else:
            print("[diag] bank never reached full capacity; no full-bank eviction comparisons recorded.")

        if mix_diag_batches:
            print(f"\n=== Mixture posterior diagnostics ({dataset_name}) ===")
            gamma_all = torch.cat([b['gamma'] for b in mix_diag_batches])
            _report_stats('gamma(z)', gamma_all)
            _report_stats('P_L.max', torch.cat([b['P_L_max'] for b in mix_diag_batches]))
            _report_stats('P_S.max', torch.cat([b['P_S_max'] for b in mix_diag_batches]))
            _report_stats('post.max', torch.cat([b['post_max'] for b in mix_diag_batches]))
            _report_stats('D_M (mixture batch)', torch.cat([b['D_M'] for b in mix_diag_batches]))
            if args.var_aligned_kappa and mix_diag_batches and 'kappa_c' in mix_diag_batches[0]:
                valid_kappa = torch.cat([b['kappa_c'] for b in mix_diag_batches])
                valid_kappa = valid_kappa[~torch.isnan(valid_kappa)]
                if valid_kappa.numel() > 0:
                    _report_stats('kappa_c (MLE, used in vMF kernel)', valid_kappa)
                if 'R_c' in mix_diag_batches[0]:
                    _report_stats('R_c (mean resultant length)',
                                  torch.cat([b['R_c'] for b in mix_diag_batches]))
                    _report_stats('weight_c (soft-label effective weight)',
                                  torch.cat([b['weight_c'] for b in mix_diag_batches]))
                    raw = torch.cat([b['kappa_raw_c'] for b in mix_diag_batches])
                    raw = raw[~torch.isnan(raw)]
                    if raw.numel() > 0:
                        _report_stats('kappa_raw_c (untruncated MLE)', raw)
                if 'log_C_c' in mix_diag_batches[0]:
                    log_c = torch.cat([b['log_C_c'] for b in mix_diag_batches])
                    log_c = log_c[~torch.isnan(log_c)]
                    if log_c.numel() > 0:
                        _report_stats('log_C_c (vMF normalizer per class)', log_c)

    if cache_tsne_viz:
        retained_idx = cache_sid[cache_sid >= 0].detach().cpu().numpy()
        out = getattr(args, 'cache_tsne_output', None) or (
            f'scripts/figures/cache_tsne_{dataset_name}_bs{args.bank_size}.png')
        render_cache_tsne(
            torch.stack(tsne_feats).numpy(),
            np.array(tsne_labels, dtype=np.int64),
            retained_idx, dataset_name, out, args,
            class_means=mean.detach().cpu().numpy())

    return sum(accuracies) / len(accuracies)


def main_worker(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print("=> Model created: visual backbone {} on {}".format(args.arch, device))
    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.arch, device=device)
    clip_model.eval()

    datasets = args.test_set.split('/')
    date = datetime.now().strftime("%b%d_%H-%M-%S")
    group_name = f"{args.arch}_Online_{date}"
    args_dict_param = {k: v for k, v in vars(args).items() if k != 'test_set'}
    all_results = {}

    for dataset_name in datasets:
        print("Extracting features for: {}".format(dataset_name))

        args.scale = custom_scale[dataset_name]

        # ============================clip_weights  ============================
        if args.GPT:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/GPT_w_{args.class_type}_class_emb"
        else:
            if args.class_type not in ["Ensemble", "Img_temp", "Custom", "Vanilla"]:
                raise NotImplementedError
            clip_weights_dir = f"./pre_extracted_class_feat/{args.arch.replace('/', '')}/{args.class_type}_class_emb"

        clip_weights = torch.load(
            os.path.join(clip_weights_dir, f"{dataset_name}.pth"), map_location=device)

        run_name = f"{dataset_name}_Online"
        run = wandb.init(project=WANDB_PROJECT, config=args_dict_param, group=group_name, name=run_name)
        val_loader = build_test_loader(dataset_name, preprocess, args.data, batch_size=args.batch_size)

        acc = evaluation(val_loader, clip_weights, clip_model.encode_image, args, dataset_name)
        all_results[dataset_name] = acc
        wandb.log({f"{dataset_name}": acc})
        run.finish()

    if all_results:
        avg_acc = sum(all_results.values()) / len(all_results)
        print("\n=== Evaluation Summary ===")
        for name, acc in all_results.items():
            print(f"{name}: {acc:.2f}")
        print(f"Average: {avg_acc:.2f}")

        summary_run = wandb.init(
            project=WANDB_PROJECT,
            config={**args_dict_param, "per_dataset_results": all_results},
            group=group_name,
            name="Average_Online",
        )
        summary_log = {name: acc for name, acc in all_results.items()}
        summary_log["Average"] = avg_acc
        wandb.log(summary_log)
        wandb.summary["Average"] = avg_acc
        summary_run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vMFcache: cache-DQDA + vMF mixture online TTA')
    parser.add_argument('--data', metavar='DIR', default='/home/liangyiwen/datasets', help='path to dataset root')
    parser.add_argument('--test_set', type=str, default='eurosat', help='dataset name (FG datasets only)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help=" CLIP model backbone:'RN50' or'ViT-B/16'.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bank_size', type=int, default=16, help="Bank Size L")
    parser.add_argument('--alpha', type=float, default=0.9, help="the alpha for EMA")
    parser.add_argument('--batch_size', type=int, default=1, help='DataLoader batch size for online evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')

    ### Bayesian mixture hyper-parameters
    parser.add_argument('--kappa_tau', type=float, default=0.01,
                        help='fixed-kappa path: vMF concentration kappa=1/kappa_tau')
    parser.add_argument('--ps_temperature', type=float, default=175.0,
                        help='var_aligned_kappa path: P_S = softmax(logpS / ps_temperature)')
    parser.add_argument('--var_aligned_kappa', action='store_true', default=False,
                        help='per-class MLE kappa from soft-label dispersion + ps_temperature calibration')
    parser.add_argument('--kappa_disp_min_weight', type=float, default=2.0,
                        help='min soft-label weight before MLE kappa for a class (else kappa_fallback)')
    parser.add_argument('--kappa_fallback', type=float, default=2000.0,
                        help='var_aligned_kappa: kappa for classes with insufficient soft-label weight')
    parser.add_argument('--eta', type=float, default=0.75, help="gamma(z) scale factor")
    parser.add_argument('--rho', type=float, default=2.0, help="gamma(z) exponent")
    parser.add_argument('--chi2_low', type=float, default=0.0, help="lower empirical quantile for safe annulus")
    parser.add_argument('--chi2_high', type=float, default=0.95, help="upper empirical quantile for safe annulus")
    parser.add_argument('--annulus_min_samples', type=int, default=200,
                        help="min D_M samples before empirical annulus gate activates")
    parser.add_argument('--no_delta_high_gate', action='store_true', default=True,
                        help="disable upper D_M bound in admission gate (default: True / none)")
    parser.add_argument('--use_delta_high_gate', action='store_false', dest='no_delta_high_gate',
                        help="enable upper D_M bound using --chi2_high (overrides default none)")
    parser.add_argument('--lambda_div', type=float, default=1.0,
                        help='diversity penalty weight on max pairwise cosine similarity')
    parser.add_argument('--clip_weight', type=float, default=1.0, help="CLIP zero-shot ensemble weight")
    parser.add_argument('--diag_stats', action='store_true', default=False, help="collect entropy/redundancy admission statistics")
    parser.add_argument('--cache_tsne_viz', action='store_true', default=False,
                        help='after evaluation, t-SNE plot of cache vs all test features')
    parser.add_argument('--cache_tsne_output', type=str, default=None,
                        help='PNG path (default scripts/figures/cache_tsne_<dataset>_bs<L>.png)')
    parser.add_argument('--cache_tsne_max_classes', type=int, default=10,
                        help='plot top-N classes by retained count (0=all)')
    parser.add_argument('--cache_tsne_max_bg', type=int, default=4000,
                        help='subsample background points for t-SNE speed')
    parser.add_argument('--cache_tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--cache_tsne_iter', type=int, default=1000)
    parser.add_argument('--cache_tsne_bg_alpha', type=float, default=0.45,
                        help='scatter alpha for non-cache points')

    ### class embedding
    parser.add_argument('--class_type', default='Custom', type=str, help=" Type of the initialization of mean matrix: Custom, Vanilla, Img_temp, Ensemble")
    parser.add_argument('--GPT', action='store_true', default=False, help="use the description or not ")

    args = parser.parse_args()
    set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    main_worker(args)
