"""Shared t-SNE visualization: retained cache vs all test features."""
import os

import matplotlib.pyplot as plt
import numpy as np


def assign_cache_sid(cache, cache_sid, pred, bank_size, global_idx, before_slot_vecs, existing_count_before):
    """Record which global test index occupies the updated bank slot."""
    start_idx = pred * bank_size
    if existing_count_before < bank_size:
        insert_local = existing_count_before
    else:
        after = cache[0][start_idx:start_idx + bank_size]
        insert_local = int((after - before_slot_vecs).abs().sum(dim=1).argmax().item())
    cache_sid[start_idx + insert_local] = global_idx


def select_classes(feats, labels, retained_idx, max_classes, seed):
    if max_classes is None or max_classes <= 0:
        return feats, labels, retained_idx, np.unique(labels)
    counts = {}
    for i in retained_idx:
        c = labels[i]
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        rng = np.random.default_rng(seed)
        classes = rng.choice(np.unique(labels), size=min(max_classes, len(np.unique(labels))), replace=False)
    else:
        classes = np.array(sorted(counts, key=counts.get, reverse=True)[:max_classes])
    cls_set = set(classes.tolist())
    keep = np.array([l in cls_set for l in labels])
    idx_map = -np.ones(labels.max() + 1, dtype=np.int64)
    for new_id, c in enumerate(classes):
        idx_map[c] = new_id
    new_labels = idx_map[labels[keep]]
    old_to_new = np.cumsum(keep) - 1
    new_retained = [old_to_new[i] for i in retained_idx if keep[i]]
    return feats[keep], new_labels, np.array(new_retained, dtype=np.int64), classes


def subsample_background(feats, labels, retained_idx, max_bg, seed):
    retained_set = set(retained_idx.tolist())
    bg_idx = [i for i in range(len(labels)) if i not in retained_set]
    if max_bg is None or len(bg_idx) <= max_bg:
        return feats, labels, retained_idx
    rng = np.random.default_rng(seed)
    pick_bg = rng.choice(bg_idx, size=max_bg, replace=False)
    keep_idx = np.sort(np.concatenate([pick_bg, retained_idx]))
    idx_map = {old: new for new, old in enumerate(keep_idx)}
    return feats[keep_idx], labels[keep_idx], np.array([idx_map[i] for i in retained_idx])


def plot_tsne(emb, labels, retained_idx, class_names, title, output_path,
              point_size_bg=10, point_size_ret=16, bg_alpha=0.45,
              mean_emb=None, mean_labels=None):
    retained_set = set(retained_idx.tolist())
    is_ret = np.array([i in retained_set for i in range(len(labels))])
    classes = np.unique(labels)
    cmap = plt.get_cmap('tab20', max(len(classes), 1))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    for ci, c in enumerate(classes):
        mask_bg = (labels == c) & (~is_ret)
        mask_ret = (labels == c) & is_ret
        color = cmap(ci % 20)
        name = class_names[c] if class_names and c < len(class_names) else f'class {c}'
        if mask_bg.any():
            ax.scatter(emb[mask_bg, 0], emb[mask_bg, 1], c=[color], s=point_size_bg,
                       alpha=bg_alpha, linewidths=0, rasterized=True)
        if mask_ret.any():
            ax.scatter(emb[mask_ret, 0], emb[mask_ret, 1], c=[color], s=point_size_ret,
                       alpha=0.95, linewidths=0.2, edgecolors='k', rasterized=True,
                       label=name if len(classes) <= 12 else None)
        if mean_emb is not None and mean_labels is not None:
            mi = np.where(mean_labels == c)[0]
            if len(mi):
                ax.scatter(mean_emb[mi, 0], mean_emb[mi, 1], c=[color], s=220,
                           marker='*', edgecolors='k', linewidths=0.8, zorder=5)
    ax.scatter([], [], c='gray', s=point_size_bg, alpha=bg_alpha, label='All test samples')
    ax.scatter([], [], c='black', s=point_size_ret, alpha=0.9, label='Retained (cache)')
    if mean_emb is not None:
        ax.scatter([], [], c='gold', s=220, marker='*', edgecolors='k', linewidths=0.8,
                   label='Class mean (μ)')
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7 if len(classes) <= 12 else 8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'[cache_tsne] saved {output_path}')


def render_cache_tsne(feats, labels, retained_idx, dataset_name, output_path, args,
                      class_means=None):
    """Fit t-SNE and save cache vs test scatter plot.

    class_means: optional [K, D] DQDA/text class centroids; appended before t-SNE so
    mean locations share the same 2D embedding as the points.
    """
    from sklearn.manifold import TSNE

    try:
        from data.cls_to_names import get_classnames
        all_names = get_classnames(dataset_name)
    except Exception:
        all_names = None

    max_classes = getattr(args, 'cache_tsne_max_classes', 10)
    max_bg = getattr(args, 'cache_tsne_max_bg', 4000)
    seed = getattr(args, 'seed', 0)
    feats, labels, retained_idx, kept_classes = select_classes(
        feats, labels, retained_idx, max_classes, seed)
    feats, labels, retained_idx = subsample_background(feats, labels, retained_idx, max_bg, seed)

    mean_feats = None
    mean_labels = None
    if class_means is not None:
        class_means = np.asarray(class_means, dtype=np.float32)
        mean_feats = class_means[kept_classes]
        mean_labels = np.arange(len(kept_classes), dtype=np.int64)

    fit_feats = feats
    if mean_feats is not None:
        fit_feats = np.vstack([feats, mean_feats])

    class_names = None
    if all_names is not None:
        class_names = [all_names[int(c)] for c in kept_classes]

    print(f'[cache_tsne] fitting on {len(fit_feats)} points '
          f'(test={len(feats)}, retained={len(retained_idx)}, means={0 if mean_feats is None else len(mean_feats)}) ...')
    perp = min(getattr(args, 'cache_tsne_perplexity', 30.0), max(5, len(fit_feats) // 4))
    tsne = TSNE(
        n_components=2, perplexity=perp,
        max_iter=getattr(args, 'cache_tsne_iter', 1000),
        init='pca', learning_rate='auto', random_state=seed)
    emb_all = tsne.fit_transform(fit_feats)
    emb = emb_all[:len(feats)]
    mean_emb = emb_all[len(feats):] if mean_feats is not None else None
    title = f'ADAPT cache (bs={args.bank_size}, {dataset_name})'
    plot_tsne(emb, labels, retained_idx, class_names, title, output_path,
              bg_alpha=getattr(args, 'cache_tsne_bg_alpha', 0.45),
              mean_emb=mean_emb, mean_labels=mean_labels)
    print(f'[cache_tsne] n_test={feats.shape[0]} n_retained={len(retained_idx)} dataset={dataset_name}')
