"""Shared training / evaluation harness for FlowLens and the baselines.

    python -m flowlens.train --model flow --data data/synthetic_50k.pt --epochs 18
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import LensDataset, load_splits
from .models import DEFAULT_LR, build_model
from .physics import ForwardModel, LensConfig

ACC_TOL_UM = 0.05  # "accurate within 1 decimal point in um" -> |error| < 0.05 um, per cell
# The synthetic diameters span only 0.30 um, so +-50 nm saturates quickly; stricter tolerances are also reported.
EXTRA_TOLS_NM = (20, 10)


def design_metrics(pred_um: torch.Tensor, true_um: torch.Tensor) -> dict:
    err = (pred_um - true_um).abs()
    out = {"acc": (err < ACC_TOL_UM).float().mean().item()}
    for nm in EXTRA_TOLS_NM:
        out[f"acc_{nm}nm"] = (err < nm * 1e-3).float().mean().item()
    out["mae_um"] = err.mean().item()
    return out


@torch.no_grad()
def evaluate(model, val: LensDataset, device, batch: int = 500, n_samples: int = 1, temperature: float = 0.0,
             rerank: bool = False, blob: dict | None = None) -> dict:
    """Per-cell accuracy and MAE (um) on the validation set.

    With n_samples > 1 and rerank=True, each candidate design is passed through
    the forward model and the candidate whose simulated response is closest
    to the target is kept ("best-of-K", with the T=0 point estimate as an extra
    candidate). This needs the synthetic `blob` for its config and waveguide flags.
    """
    model.eval()
    fwd = ForwardModel(LensConfig(**blob["config"])) if rerank else None
    preds = []
    for s in range(0, len(val), batch):
        c = val.responses[s : s + batch].to(device)
        d = model.predict(c, n_samples=n_samples, temperature=temperature)  # (B, S, 1, N, N)
        if rerank:
            # the point estimate (T=0) is always candidate 0, so reranking never loses to it
            d = torch.cat([model.predict(c, n_samples=1, temperature=0.0), d], dim=1)
            idx = val.indices[s : s + batch]
            wg = blob["meta"]["waveguide"][idx].to(device)
            target = blob["responses"][idx].float().to(device)
            d_um = val.to_um(d.clamp(-1, 1))[:, :, 0]
            b, k = d_um.shape[:2]
            sim = fwd(d_um.flatten(0, 1).double(), wg.repeat_interleave(k), noise=False)
            err = ((sim.view(b, k, *sim.shape[1:]) - target[:, None]) ** 2).flatten(2).mean(2)
            d = d[torch.arange(b), err.argmin(1)]
        else:
            d = d.mean(1)
        preds.append(val.to_um(d.clamp(-1, 1)).cpu())
    model.train()
    return design_metrics(torch.cat(preds), val.designs)


def train(model_name: str, data: str, epochs: int, out_dir: str, train_frac: float = 1.0, batch_size: int = 128,
          lr: float | None = None, seed: int = 0, device: str | None = None, model_kwargs: dict | None = None,
          log_every: int = 50) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    train_set, val_set, blob = load_splits(data, train_frac=train_frac, seed=seed)
    n_cells = train_set.designs.shape[-1]
    in_ch = train_set.responses.shape[1]
    model = build_model(model_name, n_cells, in_ch, **(model_kwargs or {})).to(device)
    lr = lr or DEFAULT_LR[model_name]
    model.configure(lr)
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"{model_name}_frac{train_frac:g}_seed{seed}"
    print(f"[{tag}] {n_params / 1e6:.2f}M params, {len(train_set)} train / {len(val_set)} val, lr {lr}")

    # whole training split lives on the GPU; it is small (~1.5 GB for 40k x 48^2)
    C, D = train_set.responses.to(device), train_set.designs_norm.to(device)
    history = []
    t_train = 0.0
    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(C), device=device)
        losses = []
        for s in range(0, len(C) - batch_size + 1, batch_size):
            idx = perm[s : s + batch_size]
            logs = model.train_step(C[idx], D[idx])
            losses.append(logs["loss"])
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step} " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()), flush=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t_train += time.time() - t0
        metrics = evaluate(model, val_set, device)
        rec = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "train_time_s": t_train, **metrics}
        history.append(rec)
        print(f"[{tag}] epoch {epoch}: loss {rec['train_loss']:.4f}  val acc {metrics['acc']:.4f}  "
              f"acc@10nm {metrics['acc_10nm']:.4f}  MAE {metrics['mae_um'] * 1e3:.1f} nm  "
              f"({t_train / 60:.1f} min)", flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"model": model_name, "train_frac": train_frac, "seed": seed, "n_train": len(train_set),
              "n_params": n_params, "lr": lr, "epochs": epochs, "history": history, "final": history[-1]}
    torch.save({"state_dict": model.state_dict(), "stats": train_set.stats, "model": model_name,
                "model_kwargs": model_kwargs or {}, "n_cells": n_cells, "in_ch": in_ch,
                "d_range": (train_set.d_min, train_set.d_max)}, out / f"{tag}.pt")
    (out / f"{tag}.json").write_text(json.dumps(result, indent=1))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["flow", "vae", "gan", "segmentation"], required=True)
    p.add_argument("--data", default="data/synthetic_50k.pt")
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--frac", type=float, default=1.0, help="fraction of the training split to use")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    train(a.model, a.data, a.epochs, a.out, a.frac, a.batch_size, a.lr, a.seed)


if __name__ == "__main__":
    main()
