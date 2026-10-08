# FlowLens: Flow-based Generative Modeling for Inverse Design of Flat Lens

## How to run

```bash
python3.11 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m flowlens.data --n 50000 --out data/synthetic_50k.pt
.venv/bin/python -m flowlens.experiments all --data data/synthetic_50k.pt --out runs/paper --epochs 18
```

## Data

This experiment uses synthetic data, not the FDTD simulations from the paper, so the numbers
differ from the paper's.

## Results

Mean absolute error of the predicted pillar diameter on 10,000 validation designs, after 18 epochs
on 40,000 training designs:

| Model | MAE (nm) |
|---|---|
| **FlowLens** | **3.8** |
| GAN | 4.5 |
| Segmentation CNN | 8.6 |
| VAE | 14.5 |
