"""
=============================================================================
Self-Pruning Neural Network — Tredence AI Engineering Intern Case Study
=============================================================================

Problem:  Train a feed-forward network on CIFAR-10 that learns to prune its
          own weights during training via learnable sigmoid gates + L1
          sparsity regularisation.

Structure
---------
  Part 1 : PrunableLinear  — custom layer with gate_scores parameter
  Part 2 : Sparsity loss   — L1 penalty on gate values
  Part 3 : Training loop   — combined loss, evaluation, λ comparison
  Part 4 : Reporting       — sparsity stats + gate distribution plot

Run
---
  pip install torch torchvision matplotlib
  python self_pruning_network.py

CIFAR-10 is downloaded automatically on first run (~170 MB).
If no internet is available, a synthetic dataset with the same shape
(3×32×32 images, 10 classes) is used instead, with a printed warning.
=============================================================================
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # works without a display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


# ===========================================================================
# PART 1 — PrunableLinear layer
# ===========================================================================

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable *gate* for every
    weight in the matrix.

    Parameters
    ----------
    in_features  : int
    out_features : int

    Extra parameter
    ---------------
    gate_scores : Tensor[out_features, in_features]
        Raw (unconstrained) gate parameters updated by the optimiser.
        Passed through sigmoid to produce gates in (0, 1).

    Forward pass
    ------------
        gates        = sigmoid(gate_scores)        -- squash to (0, 1)
        pruned_w     = weight * gates              -- element-wise mask
        output       = x @ pruned_w.T + bias       -- standard affine step

    Gradient flow
    -------------
    Both `weight` and `gate_scores` are nn.Parameter objects, so PyTorch's
    autograd tracks every operation above and backpropagates gradients to
    both tensors automatically. No custom backward is needed.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # --- Standard parameters -------------------------------------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        # Kaiming uniform init — same as nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # --- Gate scores (one scalar per weight) ----------------------------
        # Initialised to 0  =>  sigmoid(0) = 0.5, all gates start half-open.
        # Training drives unimportant gates toward -inf (gate -> 0)
        # while keeping important gates positive (gate stays > 0).
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — constrain gate_scores to (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)           # shape: (out, in)

        # Step 2 — mask each weight by its gate (element-wise product)
        pruned_weights = self.weight * gates              # shape: (out, in)

        # Step 3 — standard linear transform using the masked weights
        #   x             : (batch, in_features)
        #   pruned_weights : (out_features, in_features)
        return x @ pruned_weights.t() + self.bias        # (batch, out_features)

    # -----------------------------------------------------------------------
    # Helper utilities

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values.

        Because sigmoid output is always positive, |gate| = gate, so this is
        simply the sum of all gate values across this layer.
        """
        return torch.sigmoid(self.gate_scores).sum()

    def sparsity_level(self, threshold: float = 0.05) -> float:
        """Fraction of weights whose gate value is below `threshold`."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            return (gates < threshold).float().mean().item()

    def gate_values(self) -> np.ndarray:
        """Return all gate values as a flat NumPy array (for plotting)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).cpu().numpy().flatten()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ===========================================================================
# Network definition — feed-forward classifier using PrunableLinear layers
# ===========================================================================

class SelfPruningNet(nn.Module):
    """
    3-hidden-layer feed-forward network for CIFAR-10.

    Input : 3 x 32 x 32 images -> flattened to 3072-dimensional vectors.
    Output: 10 class logits.

    Every Linear operation uses PrunableLinear so every weight can be gated.
    BatchNorm is applied after each hidden layer for training stability.
    Dropout is applied before gating kicks in hard, acting as a warm
    regulariser while weights are still establishing their usefulness.

    Architecture (wider than minimal to give the pruner room to work):
        3072 -> 1024 -> 512 -> 256 -> 10
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024,  512)
        self.fc3 = PrunableLinear(512,   256)
        self.fc4 = PrunableLinear(256,    10)   # output layer

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        self.drop = nn.Dropout(p=dropout)

        # Collect prunable layers for easy access
        self._prunable = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)              # (B, 3, 32, 32) -> (B, 3072)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

    # ------------------------------------------------------------------
    # PART 2 — Sparsity regularisation loss
    # ------------------------------------------------------------------

    def total_sparsity_loss(self) -> torch.Tensor:
        """
        Sum the L1 gate penalty across ALL PrunableLinear layers.

        SparsityLoss = sum over all layers of sum_ij sigmoid(gate_score_ij)

        Total Loss = CrossEntropy + lambda * SparsityLoss

        This is minimised when all gates -> 0 (fully sparse network).
        Lambda controls how strongly this competes with the classification loss.
        """
        return sum(layer.sparsity_loss() for layer in self._prunable)

    # ------------------------------------------------------------------
    # Reporting helpers

    def overall_sparsity(self, threshold: float = 0.05) -> float:
        """Average fraction of pruned weights across all layers."""
        levels = [l.sparsity_level(threshold) for l in self._prunable]
        return float(np.mean(levels))

    def all_gate_values(self) -> np.ndarray:
        """Concatenated gate values from every layer (used for plotting)."""
        return np.concatenate([l.gate_values() for l in self._prunable])

    def layer_sparsity_report(self, threshold: float = 0.05) -> dict:
        names = ["fc1", "fc2", "fc3", "fc4"]
        return {name: layer.sparsity_level(threshold)
                for name, layer in zip(names, self._prunable)}

    @property
    def weight_counts(self) -> dict:
        """Number of weights per layer — derived from actual layer dims."""
        return {name: layer.in_features * layer.out_features
                for name, layer in zip(["fc1","fc2","fc3","fc4"], self._prunable)}


# ===========================================================================
# PART 3 — Data loading
# ===========================================================================

def get_cifar10_loaders(batch_size: int = 256):
    """
    Download and return CIFAR-10 train/test DataLoaders.
    Falls back to synthetic data if the download fails (e.g. no internet).
    """
    try:
        import torchvision
        import torchvision.transforms as transforms

        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)

        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True,  download=True, transform=train_tf)
        test_set  = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_tf)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
        test_loader  = DataLoader(
            test_set,  batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True)

        print("Using real CIFAR-10 dataset.")
        return train_loader, test_loader, True

    except Exception as e:
        print(f"WARNING: Could not load CIFAR-10 ({e}).")
        print("Falling back to synthetic data (same shape: 3x32x32, 10 classes).")
        print("The self-pruning mechanism is identical; accuracy numbers will differ.\n")
        return _make_synthetic_loaders(batch_size)


def _make_synthetic_loaders(batch_size: int):
    """
    Synthetic dataset that mimics CIFAR-10 shape (3x32x32, 10 classes).
    Only the first 200 of 3072 features carry class signal, making the
    sparsity-vs-accuracy trade-off clearly visible even without real images.
    """
    def _make(n_samples, seed):
        torch.manual_seed(seed)
        X = torch.randn(n_samples, 3, 32, 32)
        # Class determined by argmax over the first 200 pixels
        flat = X.view(n_samples, -1)
        y = (flat[:, :200].argmax(dim=1) % 10)
        return TensorDataset(X, y)

    train_loader = DataLoader(
        _make(10000, seed=0), batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(
        _make(2000,  seed=1), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, False


# ===========================================================================
# PART 3 — Training and evaluation
# ===========================================================================

def build_optimizer(model: SelfPruningNet, lr_weights: float, lr_gates: float):
    """
    Two separate param groups:

      1. weights + biases  — standard LR + L2 weight decay
      2. gate_scores       — higher LR, NO weight decay

    Gate scores are excluded from weight decay because the sparsity loss
    already penalises them via L1. Mixing L1 and L2 on the same parameter
    produces muddled gradients and slows convergence.

    Gate scores need a higher LR because sigmoid saturates: to drive
    sigmoid(gs) from 0.5 down to near 0, the score must reach roughly -5.
    A higher LR lets the gates travel this distance in fewer epochs.
    """
    gate_param_ids = {id(l.gate_scores) for l in model._prunable}

    weight_params = [p for p in model.parameters() if id(p) not in gate_param_ids]
    gate_params   = [p for p in model.parameters() if id(p)     in gate_param_ids]

    return torch.optim.Adam([
        {"params": weight_params, "lr": lr_weights, "weight_decay": 1e-4},
        {"params": gate_params,   "lr": lr_gates,   "weight_decay": 0.0},
    ])


def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    """
    One full pass over the training set computing:

        Total Loss = CrossEntropyLoss  +  lambda * SparsityLoss

    Both terms are differentiable, so a single backward() pass computes
    gradients for weights, biases, AND gate_scores simultaneously.
    """
    model.train()
    total_ce = total_sp = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        ce_loss = F.cross_entropy(logits, labels)    # classification term
        sp_loss = model.total_sparsity_loss()        # L1 gate penalty

        loss = ce_loss + lambda_sparse * sp_loss     # combined loss (Part 2)

        loss.backward()      # gradients flow to weight AND gate_scores
        optimizer.step()

        total_ce += ce_loss.item()
        total_sp += sp_loss.item()

    n = len(loader)
    return total_ce / n, total_sp / n


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return top-1 accuracy on the given DataLoader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def run_experiment(
    lambda_sparse:      float,
    epochs:             int,
    device:             torch.device,
    train_loader,
    test_loader,
    lr_weights:         float = 1e-3,
    lr_gates:           float = 3e-3,
    warmup_epochs:      int   = 5,
    sparsity_threshold: float = 0.05,
) -> dict:
    """
    Train one self-pruning network for `epochs` epochs with the given lambda.

    Warmup phase (first `warmup_epochs` epochs):
        Gate scores are frozen — only weights are trained.
        This lets the network first learn useful representations before
        the sparsity penalty starts eliminating connections.
        Without warmup, high-LR gates prune weights before they've had
        a chance to prove their value, hurting final accuracy.

    Pruning phase (remaining epochs):
        Gates are unfrozen and updated by the full combined loss.

    Returns accuracy, overall sparsity, per-layer sparsity, and gate values.
    """
    model     = SelfPruningNet().to(device)
    optimizer = build_optimizer(model, lr_weights, lr_gates)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n{'─'*60}")
    print(f"  Experiment: lambda = {lambda_sparse:.0e}   ({epochs} epochs, "
          f"warmup={warmup_epochs})")
    print(f"{'─'*60}")

    for epoch in range(1, epochs + 1):

        # ── Warmup: freeze gate_scores so only weights learn ──────────────
        if epoch <= warmup_epochs:
            for layer in model._prunable:
                layer.gate_scores.requires_grad_(False)
            effective_lambda = 0.0          # no sparsity loss during warmup
        else:
            for layer in model._prunable:
                layer.gate_scores.requires_grad_(True)
            effective_lambda = lambda_sparse

        ce, sp = train_one_epoch(
            model, train_loader, optimizer, device, effective_lambda)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sp_level = model.overall_sparsity(sparsity_threshold)
            phase    = "warmup" if epoch <= warmup_epochs else "pruning"
            print(f"  ep {epoch:3d}/{epochs} [{phase:7s}]  |  "
                  f"CE={ce:.4f}  SpLoss={sp:.0f}  |  "
                  f"TestAcc={acc:.3f}  Sparsity={sp_level*100:.1f}%")

    # Final evaluation
    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.overall_sparsity(sparsity_threshold)
    layer_report   = model.layer_sparsity_report(sparsity_threshold)
    gate_vals      = model.all_gate_values()
    wc             = model.weight_counts

    print(f"\n  FINAL  Accuracy={final_acc*100:.2f}%  "
          f"Overall Sparsity={final_sparsity*100:.1f}%")
    for layer_name, sp in layer_report.items():
        n_w    = wc[layer_name]
        pruned = int(sp * n_w)
        print(f"    {layer_name}: {sp*100:.1f}% pruned "
              f"({pruned:,} / {n_w:,} weights)")

    return {
        "lambda":   lambda_sparse,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates":    gate_vals,
    }


# ===========================================================================
# PART 3 — Gate distribution plot (required deliverable)
# ===========================================================================

def plot_gate_distributions(results: list, save_path: str = "gate_distributions.png"):
    """
    Histogram of final gate values for each lambda.

    Expected output (rubric):
      - A large spike at 0  =>  pruned (silenced) weights
      - A cluster away from 0  =>  retained (active) connections

    The bimodal shape grows more pronounced as lambda increases.
    """
    THRESHOLD = 0.05
    n         = len(results)
    palette   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, res, color in zip(axes, results, palette):
        gates  = res["gates"]
        lam    = res["lambda"]
        acc    = res["accuracy"] * 100
        sp     = res["sparsity"] * 100
        pruned = int((gates < THRESHOLD).sum())
        total  = len(gates)

        ax.hist(gates, bins=80, color=color, edgecolor="none", alpha=0.88)
        ax.axvline(THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                   label=f"Prune threshold ({THRESHOLD})")

        ax.set_title(
            f"lambda = {lam:.0e}\n"
            f"Test Accuracy = {acc:.1f}%   |   Sparsity = {sp:.1f}%",
            fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Gate value  sigmoid(gate_score)", fontsize=10)
        ax.set_ylabel("Weight count", fontsize=10)

        ax.text(0.97, 0.95,
                f"Pruned: {pruned:,} / {total:,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="darkred",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="mistyrose", alpha=0.75))
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Gate Value Distribution After Training\n"
        "Spike near 0 = pruned weights  |  "
        "Cluster > threshold = retained connections",
        fontsize=13, fontweight="bold", y=1.03)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {save_path}")


# ===========================================================================
# Main — run all three lambda experiments and print summary table
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS  = 60          # more epochs needed: 5 warmup + 55 pruning

    # Three lambda values: low / medium / high sparsity pressure.
    # With warmup, weights establish themselves before gates start pruning,
    # so we can afford slightly higher lambdas for clearer trade-off curves.
    LAMBDAS = [1e-4, 5e-4, 2e-3]

    WARMUP_EPOCHS      = 5     # freeze gates for first N epochs
    SPARSITY_THRESHOLD = 0.05  # gates below this are considered pruned

    print("=" * 60)
    print("  Self-Pruning Neural Network  |  CIFAR-10")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {EPOCHS} per experiment  (warmup={WARMUP_EPOCHS})")
    print(f"  Lambdas : {LAMBDAS}")
    print(f"  Prune threshold : {SPARSITY_THRESHOLD}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, test_loader, using_real = get_cifar10_loaders(batch_size=256)

    # -----------------------------------------------------------------------
    # Run one experiment per lambda value
    # -----------------------------------------------------------------------
    results = []
    for lam in LAMBDAS:
        torch.manual_seed(42)       # same random init for fair comparison
        res = run_experiment(
            lambda_sparse      = lam,
            epochs             = EPOCHS,
            device             = DEVICE,
            train_loader       = train_loader,
            test_loader        = test_loader,
            warmup_epochs      = WARMUP_EPOCHS,
            sparsity_threshold = SPARSITY_THRESHOLD,
        )
        results.append(res)

    # -----------------------------------------------------------------------
    # Summary table  (required in the Markdown report)
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 55)
    print(f"  {'Lambda':^10}  {'Test Accuracy':^14}  {'Sparsity Level':^14}")
    print("  " + "-" * 51)
    for r in results:
        print(f"  {r['lambda']:^10.0e}  {r['accuracy']*100:^13.2f}%  "
              f"{r['sparsity']*100:^13.1f}%")
    print("=" * 55)

    if not using_real:
        print("\n  Note: results above use SYNTHETIC data.")
        print("  Re-run with internet access to reproduce on real CIFAR-10.")

    # -----------------------------------------------------------------------
    # Gate distribution plot
    # -----------------------------------------------------------------------
    plot_gate_distributions(results, save_path="gate_distributions.png")

    print("\nSubmission files:")
    print("  self_pruning_network.py   <- this script")
    print("  gate_distributions.png    <- gate histogram plot")
    print("  report.md                 <- written analysis")