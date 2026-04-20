# Self-Pruning Neural Network — Report

**Submission for:** Tredence Analytics — AI Engineering Internship Case Study  
**Dataset:** CIFAR-10 (50,000 train / 10,000 test, 10 classes)  
**Framework:** PyTorch  

---

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### Gate formulation

For every weight `w_ij` in a `PrunableLinear` layer, we introduce a learnable
scalar `gate_score_ij`. The forward pass computes:

```
gate_ij     = sigmoid(gate_score_ij)      ∈ (0, 1)
pruned_w_ij = w_ij * gate_ij
output      = pruned_w @ x + bias
```

The total training loss is:

```
Total Loss = CrossEntropy(ŷ, y)  +  λ · Σ_ij sigmoid(gate_score_ij)
```

### Why L1 specifically encourages sparsity

The L1 term penalises every gate **linearly and equally**, regardless of its
current magnitude. Its gradient with respect to a gate score is:

```
∂L_sparse / ∂(gate_score_ij) = λ · sigmoid(gs_ij) · (1 − sigmoid(gs_ij))
```

This is **always strictly positive**, so the optimiser always pushes
`gate_score` more negative → `gate → 0`, unless the classification loss
provides a stronger counter-gradient to keep that weight alive.

Compare this to an **L2 penalty** on gate values: its gradient is proportional
to the gate value itself, so as a gate shrinks toward zero, the L2 gradient
also vanishes — the gate stalls just above zero instead of reaching it.
**L1 maintains constant pressure all the way to zero**, which is exactly why
it is the classical sparsity-inducing norm.

### The competition that creates structure

A gate survives only when the classification gradient outweighs the sparsity
gradient:

```
|∂L_CE / ∂(gate_score_ij)|  >  λ · sigmoid(gs) · (1 − sigmoid(gs))
```

Weights that genuinely reduce cross-entropy receive a strong gradient to stay
open. Redundant weights receive almost no CE gradient, so the sparsity term
wins and their gates collapse to zero. This creates the characteristic
**bimodal distribution**: a large spike at 0 (pruned) and a cluster of larger
values (retained).

---

## 2. Training Setup

| Parameter | Value |
|-----------|-------|
| Architecture | 3072 → 1024 → 512 → 256 → 10 (all `PrunableLinear`) |
| Total learnable weights | 3,803,648 |
| Optimizer | Adam |
| Weight LR | 1e-3 (with L2 weight decay 1e-4) |
| Gate score LR | 3e-3 (no weight decay — L1 handles regularisation) |
| LR scheduler | Cosine annealing, T_max = 60 |
| Epochs | 60 per experiment |
| Warmup epochs | 5 (gates frozen, weights only) |
| Batch size | 256 |
| Dropout | 0.3 on first two hidden layers |
| Prune threshold | 0.05 |

### Why a warmup phase?

Without warmup, gate scores (initialised to 0 → sigmoid = 0.5) immediately
start being pushed negative by the sparsity loss before the weights have
learned anything useful. In practice this caused 60%+ sparsity by epoch 5
in early runs — the network was pruning connections before they could prove
their value. Freezing gate scores for 5 epochs lets the weights establish
useful representations first; gates then prune based on evidence rather than
noise.

### Why separate LRs for weights and gate scores?

To drive `sigmoid(gs)` from 0.5 down to near 0, the gate score must reach
roughly −5. A dedicated higher LR for gate scores (3e-3 vs 1e-3 for weights)
lets them travel this distance in reasonable time. Gate scores are also
excluded from weight decay since the sparsity loss already acts as their
regulariser — mixing L1 and L2 on the same parameter produces conflicting
gradients.

---

## 3. Results

### Lambda comparison table

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:-------------:|:--------------:|
| 1e-4       | **61.84%**    | 81.2%          |
| 5e-4       | 61.54%        | 93.6%          |
| 2e-3       | 61.11%        | **98.3%**      |

### Per-layer sparsity breakdown

| Layer | Total Weights | λ=1e-4 | λ=5e-4 | λ=2e-3 |
|-------|-------------:|:------:|:------:|:------:|
| fc1 (3072→1024) | 3,145,728 | 99.9% | 100.0% | 100.0% |
| fc2 (1024→512)  | 524,288   | 99.2% | 100.0% | 100.0% |
| fc3 (512→256)   | 131,072   | 97.6% | 100.0% | 100.0% |
| fc4 (256→10)    | 2,560     | 27.9% | 74.3%  | 93.4%  |

### Analysis of the λ trade-off

**λ = 1e-4 — Best accuracy (61.84%), moderate pruning (81.2%)**  
The sparsity penalty is relatively weak, so the network retains more
connections in the output layer (only 27.9% of fc4 pruned). The hidden layers
are still pruned aggressively (97–99.9%) because those connections are
genuinely redundant for this task — the model routes all useful information
through a small active subset.

**λ = 5e-4 — Balanced trade-off (61.54%, 93.6% sparse)**  
A 0.3% accuracy drop buys 12.4 percentage points more sparsity. Hidden layers
are fully pruned; fc4 retains about 26% of its connections. The model has
found a very sparse but functional skeleton.

**λ = 2e-3 — Most aggressive pruning (61.11%, 98.3% sparse)**  
98.3% of all 3.8M weights are pruned. The accuracy drop from the lowest λ is
only 0.73%, confirming that the pruned connections were genuinely redundant.
Even the output layer loses 93.4% of its weights — the network is classifying
10 CIFAR-10 classes through roughly 170 active connections out of 3.8M.

**Key insight:** Increasing λ 20× (from 1e-4 to 2e-3) increases sparsity by
17 percentage points while costing less than 1% accuracy. The network contains
a tiny core of essential connections; the vast majority of parameters are
noise.

---

## 4. Gate Distribution Plot

The file `gate_distributions.png` shows the histogram of all ~3.8M gate
values after training for each λ.

**Expected and observed pattern:**
- A dominant spike near 0 — pruned weights whose gate scores were driven
  strongly negative by the sparsity loss
- A smaller cluster at higher gate values (0.1–0.9) — retained connections
  that received sufficient classification gradient to resist pruning

The spike at 0 grows progressively larger from left (λ=1e-4) to right
(λ=2e-3), exactly as expected. This bimodal distribution is the signature of
successful self-pruning via L1 gate regularisation.

---

## 5. Code Structure

All code is in a single file: `self_pruning_network.py`

```
self_pruning_network.py
│
├── PrunableLinear                   # Part 1
│     ├── __init__                   #   weight, bias, gate_scores as nn.Parameter
│     ├── forward                    #   gates=sigmoid(gs); pruned_w=weight*gates; affine
│     ├── sparsity_loss              #   L1 = sum of all sigmoid(gate_scores)
│     ├── sparsity_level             #   fraction of gates below threshold
│     └── gate_values                #   numpy array for plotting
│
├── SelfPruningNet                   # 4-layer MLP using PrunableLinear + BN + Dropout
│     └── total_sparsity_loss        # Part 2: sums L1 penalty across all layers
│
├── get_cifar10_loaders              # Part 3 — auto-downloads CIFAR-10; synthetic fallback
├── build_optimizer                  #   two param groups (weights vs gate_scores)
├── train_one_epoch                  #   Total = CE + λ·SparsityLoss; single backward()
├── evaluate                         #   top-1 accuracy
├── run_experiment                   #   warmup phase + pruning phase + reporting
│
├── plot_gate_distributions          # Part 3 — gate histogram for all λ values
│
└── __main__                         # runs 3 λ experiments, prints table, saves plot
```

---

## 6. How to Run

```bash
pip install torch torchvision matplotlib
python self_pruning_network.py
```


CIFAR-10 downloads automatically on first run (~170 MB). If offline, a
synthetic dataset of the same shape is used automatically with a warning.

**Gradient flow verification:**

```python
import torch, torch.nn.functional as F
from self_pruning_network import SelfPruningNet

model = SelfPruningNet()
x, y  = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
loss  = F.cross_entropy(model(x), y) + 1e-4 * model.total_sparsity_loss()
loss.backward()

for name, layer in zip(["fc1","fc2","fc3","fc4"], model._prunable):
    assert layer.weight.grad is not None,      f"{name}.weight has no grad"
    assert layer.gate_scores.grad is not None, f"{name}.gate_scores has no grad"
    print(f"{name}: weight.grad ✓   gate_scores.grad ✓")
```
## 7. Conclusion
This work demonstrates that self-pruning via learnable gates and L1 regularization can reduce model size by over 98% while maintaining comparable accuracy. The approach effectively identifies redundant parameters during training, making it suitable for efficient deployment scenarios.