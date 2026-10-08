"""Reproduce the paper's experiments: Table 1, Fig. 2a (accuracy vs epoch) and Fig. 2b (data efficiency).

    python -m flowlens.experiments run    --data data/synthetic_50k.pt --out runs/paper
    python -m flowlens.experiments rerank --data data/synthetic_50k.pt --out runs/paper
    python -m flowlens.experiments report --data data/synthetic_50k.pt --out runs/paper

`run` skips any (model, fraction) whose result JSON already exists, so it can be resumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import LensDataset, load_splits
from .models import build_model
from .train import evaluate, train

MODELS = ["flow", "vae", "gan", "segmentation"]
LABELS = {"flow": "FlowLens", "vae": "VAE", "gan": "GAN", "segmentation": "Segmentation CNN"}
# fixed categorical order from the validated reference palette (color follows the model, never its rank)
COLORS = {"flow": "#2a78d6", "vae": "#eb6834", "gan": "#1baf7a", "segmentation": "#eda100"}
MARKERS = {"flow": "o", "vae": "s", "gan": "^", "segmentation": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _tag(model, frac, seed):
    return f"{model}_frac{frac:g}_seed{seed}"


def load_model(path, device, with_checkpoint: bool = False):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    m = build_model(ck["model"], ck["n_cells"], ck["in_ch"], **ck["model_kwargs"])
    m.load_state_dict(ck["state_dict"])
    m = m.to(device).eval()
    return (m, ck) if with_checkpoint else m


def val_for(ck, blob, val):
    """Validation set normalised with the checkpoint's own training statistics."""
    return LensDataset(blob, val.indices, stats=ck["stats"])


def run(args):
    for frac in sorted(args.fracs, reverse=True):  # full data first: Table 1 / Fig. 2a are available earliest
        for m in args.models:
            if (Path(args.out) / f"{_tag(m, frac, args.seed)}.json").exists():
                print(f"skip {_tag(m, frac, args.seed)} (done)")
                continue
            train(m, args.data, args.epochs, args.out, train_frac=frac, seed=args.seed)


def rerank(args):
    """Best-of-K: sample K designs per response, keep the one whose simulated response matches best."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, blob = load_splits(args.data, seed=args.seed)
    results = {}
    for m in args.models:
        ck = Path(args.out) / f"{_tag(m, 1.0, args.seed)}.pt"
        if not ck.exists():
            continue
        model, ckpt = load_model(ck, device, with_checkpoint=True)
        v = val_for(ckpt, blob, val)
        res = {"point": evaluate(model, v, device)}
        if m != "segmentation":  # deterministic model: sampling is meaningless
            for t in args.temperatures:
                res[f"best_of_{args.k}_T{t:g}"] = evaluate(model, v, device, batch=100, n_samples=args.k,
                                                           temperature=t, rerank=True, blob=blob)
        results[m] = res
        print(m, json.dumps(res, indent=1))
    (Path(args.out) / "rerank.json").write_text(json.dumps(results, indent=1))


# --------------------------------------------------------------------------
# Report: Table 1, Fig. 2a, Fig. 2b, example designs
# --------------------------------------------------------------------------


def _style(ax, title, ylabel, xlabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def _direct_labels(ax, ends):
    """Label each line at its right end; nudge labels apart so they never overlap."""
    ends = sorted(ends, key=lambda e: e[1])
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    ys = []
    for _, y, _ in ends:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for (x, _, text), y in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out = Path(args.out)
    rep = out / "report"
    rep.mkdir(exist_ok=True)
    res = {}
    for p in out.glob("*_seed*.json"):
        r = json.loads(p.read_text())
        res[(r["model"], r["train_frac"])] = r
    models = [m for m in args.models if (m, 1.0) in res]
    metrics = [("acc", "accuracy (|err| < 50 nm, paper metric)"), ("acc_10nm", "accuracy (|err| < 10 nm)")]

    # ---- Fig. 2a: validation accuracy vs epoch (full data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, name) in zip(axes, metrics):
        ends = []
        for m in models:
            h = res[(m, 1.0)]["history"]
            x = [r["epoch"] for r in h]
            y = [r[key] for r in h]
            ax.plot(x, y, color=COLORS[m], lw=2, marker=MARKERS[m], ms=4, label=LABELS[m])
            ends.append((x[-1], y[-1], LABELS[m]))
        _style(ax, name, "validation accuracy", "epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(right=ax.get_xlim()[1] * 1.25)
        _direct_labels(ax, ends)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Fig. 2a - validation accuracy (synthetic surrogate data)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(rep / "fig2a_accuracy_vs_epoch.png", dpi=160)
    plt.close(fig)

    # ---- Fig. 2b: final accuracy vs fraction of training data
    fracs = sorted({f for (_, f) in res})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, name) in zip(axes, metrics):
        ends = []
        for m in models:
            fs = [f for f in fracs if (m, f) in res]
            y = [res[(m, f)]["final"][key] for f in fs]
            ax.plot([100 * f for f in fs], y, color=COLORS[m], lw=2, marker=MARKERS[m], ms=5, label=LABELS[m])
            ends.append((100 * fs[-1], y[-1], LABELS[m]))
        _style(ax, name, "validation accuracy", "training data used (%)")
        ax.set_xticks([100 * f for f in fracs])
        ax.set_xlim(right=150)
        _direct_labels(ax, ends)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Fig. 2b - data efficiency (synthetic surrogate data)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(rep / "fig2b_data_efficiency.png", dpi=160)
    plt.close(fig)

    # ---- Table 1 (+ extra columns)
    rr = json.loads((out / "rerank.json").read_text()) if (out / "rerank.json").exists() else {}
    vae_final = res[("vae", 1.0)]["final"]["acc_10nm"] if ("vae", 1.0) in res else None
    thr = 0.8 * vae_final if vae_final else None
    lines = ["| Model | Acc. ±50 nm (paper) | Acc. ±20 nm | Acc. ±10 nm | MAE (nm) | Params | Train time (min) |"
             + (f" Time to {100 * thr:.1f}% @±10 nm (min) |" if thr else ""),
             "|---|---|---|---|---|---|---|" + ("---|" if thr else "")]
    for m in ["segmentation", "gan", "vae", "flow"]:
        if (m, 1.0) not in res:
            continue
        r = res[(m, 1.0)]
        f = r["final"]
        row = (f"| {LABELS[m]} | {100 * f['acc']:.1f}% | {100 * f['acc_20nm']:.1f}% | {100 * f['acc_10nm']:.1f}% "
               f"| {1e3 * f['mae_um']:.1f} | {r['n_params'] / 1e6:.2f}M | {f['train_time_s'] / 60:.1f} |")
        if thr:
            hit = next((h["train_time_s"] / 60 for h in r["history"] if h["acc_10nm"] >= thr), None)
            row += f" {hit:.1f} |" if hit is not None else " not reached |"
        lines.append(row)
    table = "\n".join(lines)
    if rr:
        table += "\n\nBest-of-K with forward-model reranking (full data):\n\n| Model | Setting | Acc. ±50 nm | Acc. ±10 nm | MAE (nm) |\n|---|---|---|---|---|\n"
        for m, d in rr.items():
            for setting, f in d.items():
                table += f"| {LABELS[m]} | {setting} | {100 * f['acc']:.1f}% | {100 * f['acc_10nm']:.1f}% | {1e3 * f['mae_um']:.1f} |\n"
    (rep / "table1.md").write_text(table + "\n")
    print(table)

    # ---- example designs: response, ground truth and each model's prediction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val, blob = load_splits(args.data, seed=args.seed)
    wg = blob["meta"]["waveguide"][val.indices]
    pick = [int(torch.nonzero(~wg)[0]), int(torch.nonzero(wg)[0])]  # one parallel-fed, one waveguide-fed
    raw = blob["responses"][val.indices[pick]].float()
    preds = {}
    for m in models:
        model, ckpt = load_model(out / f"{_tag(m, 1.0, args.seed)}.pt", device, with_checkpoint=True)
        v = val_for(ckpt, blob, val)
        d = model.predict(v.responses[pick].to(device))[:, 0, 0]
        preds[m] = val.to_um(d.clamp(-1, 1)).cpu()
    true = val.designs[pick, 0]
    cols = 3 + len(models)
    fig, axes = plt.subplots(2, cols, figsize=(2.1 * cols, 4.6))
    for row, i in enumerate(range(len(pick))):
        phase = torch.atan2(raw[i, 2], raw[i, 1])
        panels = [(raw[i, 0], "Blues", None, "response |E|²"), (phase, "twilight", (-3.1416, 3.1416), "response phase"),
                  (true[i], "Blues", (0.08, 0.38), "true design (µm)")]
        panels += [(preds[m][i], "Blues", (0.08, 0.38), LABELS[m]) for m in models]
        for ax, (img, cmap, lim, title) in zip(axes[row], panels):
            im = ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None, origin="lower")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8, color=INK)
        axes[row, 0].set_ylabel(["parallel-fed", "waveguide-fed"][row], fontsize=8, color=INK2)
    fig.colorbar(im, ax=axes[:, 2:], shrink=0.8, label="pillar diameter (µm)")
    fig.savefig(rep / "examples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"report written to {rep}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "rerank", "report", "all"])
    p.add_argument("--data", default="data/synthetic_50k.pt")
    p.add_argument("--out", default="runs/paper")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=16, help="samples per response for best-of-K reranking")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 1.0])
    a = p.parse_args()
    if a.cmd in ("run", "all"):
        run(a)
    if a.cmd in ("rerank", "all"):
        rerank(a)
    if a.cmd in ("report", "all"):
        report(a)


if __name__ == "__main__":
    main()
