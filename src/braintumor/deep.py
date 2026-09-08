"""
STAGE 6b - DEEP FEATURE CLASSIFIER (tumor typing)
=================================================
Why this module exists, and why it did not exist for the binary task.

Our engineered features answer "is there a tumor" very well: 97.7% accuracy, 96.9%
sensitivity. When we pointed the same features at the harder question - "WHICH tumor" -
they reached 72.1% overall and 24% recall on glioma. That failure was not a bug. It is what
you get when the features cannot express the thing being asked about.

Naming a tumor type depends on two things our handcrafted families capture poorly:

  1. WHERE the mass sits relative to brain anatomy. Family 7 (`features.location_features`)
     now measures this explicitly - midline distance, depth inside the parenchyma, rim
     contact - and it helps. But it can only describe the ONE region the segmenter found.
  2. The internal architecture of the mass and the tissue around it: how the edge fades,
     whether there is a necrotic core, how the surrounding white matter is displaced. These
     are spatial patterns over many scales, which is precisely what a convolutional network
     learns and what a global texture histogram averages away.

So we add a CNN, and we add it as a SECOND OPINION rather than a replacement. The classical
model stays; `train.py` combines the two probability vectors. That preserves the property
that made this project defensible in the first place - a radiologist can still see the
segmented region and the measurements behind it - while letting the network contribute the
part of the judgement that the measurements cannot reach.

TRANSFER LEARNING. We do not train from scratch. A ResNet-18 pretrained on ImageNet already
encodes edges, textures, blobs and part-whole structure; those early layers transfer to MRI
even though ImageNet contains no medical images. Starting from them means we need thousands
of images rather than hundreds of thousands, which is what makes this trainable in minutes
on a laptop with no dedicated GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

# torch is an optional dependency. Everything else in the project - the whole binary
# pipeline, the CLI, the API - must keep working when it is absent, so the import is
# guarded and every entry point checks `is_available()` first.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _TORCH_ERROR: str | None = None
except Exception as exc:                                    # pragma: no cover - env specific
    torch = None                                            # type: ignore[assignment]
    nn = None                                               # type: ignore[assignment]
    _TORCH_ERROR = str(exc)


def is_available() -> bool:
    """True when PyTorch imported cleanly and the deep branch can run."""
    return torch is not None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed, so the deep tumor-typing branch is unavailable.\n"
            f"  import error: {_TORCH_ERROR}\n"
            "  fix with:     pip install torch torchvision"
        )


def pick_device() -> "torch.device":
    """Prefer Apple's Metal backend, then CUDA, then CPU.

    On an M-series laptop MPS turns a ~40-minute CPU fine-tune into a few minutes, which is
    the difference between "we trained a CNN" and "we would have, given a cluster".
    """
    require_torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ======================================================================================
# DATA
# ======================================================================================
# ImageNet channel statistics. The pretrained weights were fitted on inputs normalised with
# these constants, so we must reuse them exactly - feeding differently-scaled inputs to a
# pretrained trunk silently wastes most of the transfer.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

CNN_INPUT_SIZE = 224


class _MRIDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    """Wraps the enhanced 256x256 grayscale slices we already computed in Stages 2-3.

    Feeding the CNN our ENHANCED image rather than the raw file is deliberate. It keeps the
    deep branch downstream of the same preprocessing every other part of the system uses,
    so the network and the classical model are looking at the identical picture. It also
    means CLAHE has already removed the bias field, and the network does not have to spend
    capacity undoing scanner inhomogeneity.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray | None, train: bool,
                 seed: int = config.RANDOM_SEED):
        self.images = images
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Augmentation aimed at ONE measured problem, not sprinkled on for luck.

        We showed in docs/06_DATASETS.md that this dataset's train and test halves are not
        drawn from the same source: every training glioma is 512x512 while the test gliomas
        arrive in 64 different sizes, and a trivial classifier can tell the halves apart at
        0.858 AUC from the histogram alone. A model trained on the raw training images
        learns that uniformity and then meets something else at test time.

        Each transform below attacks one axis of that shift:
          * random resized crop  -> scale and framing differences
          * rotation / flip      -> patient positioning, and laterality which is irrelevant
                                    to tumor TYPE (a left glioma is still a glioma)
          * brightness/contrast  -> scanner and sequence differences
          * gamma                -> non-linear intensity mapping between vendors
        """
        import cv2

        h, w = img.shape[:2]

        # -- random resized crop: keep 78-100% of the frame, then scale back up
        scale = self.rng.uniform(0.78, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = int(self.rng.integers(0, h - ch + 1))
        x0 = int(self.rng.integers(0, w - cw + 1))
        img = img[y0:y0 + ch, x0:x0 + cw]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # -- rotation about the centre, +/- 15 degrees
        angle = float(self.rng.uniform(-15, 15))
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if self.rng.random() < 0.5:
            img = img[:, ::-1].copy()

        # -- photometric jitter
        alpha = float(self.rng.uniform(0.85, 1.15))          # contrast
        beta = float(self.rng.uniform(-18, 18))              # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)

        if self.rng.random() < 0.3:
            gamma = float(self.rng.uniform(0.75, 1.35))
            img = 255.0 * np.power(img / 255.0, gamma)

        return img.astype(np.uint8)

    def __getitem__(self, idx: int):
        import cv2

        img = self.images[idx]
        if self.train:
            img = self._augment(img)

        img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0

        # MRI is single-channel; ImageNet weights expect three. Replicating the grey channel
        # is the standard bridge - it keeps the pretrained first-layer filters meaningful,
        # whereas averaging them into one channel throws away colour-opponent structure the
        # later layers still expect to receive.
        x = np.stack([x, x, x], axis=0)
        for c in range(3):
            x[c] = (x[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]

        tensor = torch.from_numpy(x)
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[idx])


# ======================================================================================
# MODEL
# ======================================================================================
def build_backbone(n_classes: int, seed: int = config.RANDOM_SEED, pretrained: bool = True):
    """ResNet-18 with the 1000-way ImageNet head swapped for our n-way head.

    ResNet-18 rather than something deeper: with ~2900 training images, a larger trunk
    overfits before it generalises, and depth is not the binding constraint here - the
    domain shift is. The residual connections matter more than the parameter count, since
    they are what let gradients reach the early layers we are fine-tuning.
    """
    require_torch()
    from torchvision import models

    torch.manual_seed(seed)
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:                                    # pragma: no cover
            weights = None

    net = models.resnet18(weights=weights)
    in_features = net.fc.in_features
    # Dropout before the classifier. The trunk is pretrained and the head is random, so the
    # head is where overfitting starts; this is the cheapest place to regularise.
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, n_classes),
    )
    return net


class DeepTumorClassifier:
    """A scikit-learn-shaped wrapper around the fine-tuned CNN.

    Exposing `fit` / `predict` / `predict_proba` is not decoration: it lets the deep model
    drop into the same evaluation, ensembling and reporting code that the classical models
    already use, so the two are compared on identical terms with no special-casing.
    """

    def __init__(self, n_classes: int = 4, epochs: int = 14, batch_size: int = 32,
                 lr: float = 3e-4, head_lr: float = 1e-3, weight_decay: float = 1e-4,
                 seed: int = config.RANDOM_SEED, device: str | None = None,
                 pretrained: bool = True, verbose: bool = True,
                 tta: bool = config.DEEP_TTA):
        require_torch()
        self.tta = tta
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.verbose = verbose
        self.pretrained = pretrained
        self.device = torch.device(device) if device else pick_device()
        self.net = None
        self.history: list[dict] = []
        self.best_val_balanced_acc: float | None = None

    # ---------------------------------------------------------------- training
    def fit(self, images: np.ndarray, labels: np.ndarray,
            val_images: np.ndarray | None = None, val_labels: np.ndarray | None = None):
        """Fine-tune the network. Model selection uses `val_*` only - never the test set."""
        require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        labels = np.asarray(labels, dtype=np.int64)
        self.net = build_backbone(self.n_classes, self.seed, self.pretrained).to(self.device)

        train_ds = _MRIDataset(images, labels, train=True, seed=self.seed)
        # num_workers=0: the augmentation is cheap OpenCV work and MPS does not play well
        # with forked worker processes, so spawning them costs more than it saves here.
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)

        val_dl = None
        if val_images is not None and val_labels is not None and len(val_images):
            val_ds = _MRIDataset(val_images, np.asarray(val_labels), train=False, seed=self.seed)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # Class weighting, same reasoning as the classical models: the 4-class set is 85%
        # tumor and 15% healthy, and "no tumor" is the class we can least afford to lose.
        counts = np.bincount(labels, minlength=self.n_classes).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / (self.n_classes * np.maximum(counts, 1)), 0.0)
        class_weight = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Label smoothing of 0.05. The dataset's own labels are imperfect (its glioma split
        # is known to be inconsistent), so training the network to be 100% certain teaches
        # it to be confident about noise. Smoothing keeps the probabilities usable for the
        # ensemble, which needs calibrated inputs rather than saturated ones.
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=0.05)

        # Two parameter groups: the pretrained trunk moves slowly, the fresh head moves
        # fast. One global learning rate would either destroy the transferred features or
        # leave the head undertrained.
        head_params = list(self.net.fc.parameters())
        head_ids = {id(p) for p in head_params}
        trunk_params = [p for p in self.net.parameters() if id(p) not in head_ids]
        optimiser = torch.optim.AdamW(
            [{"params": trunk_params, "lr": self.lr},
             {"params": head_params, "lr": self.head_lr}],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)

        best_state = None
        best_score = -1.0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total, correct, loss_sum = 0, 0, 0.0
            for xb, yb in train_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                out = self.net(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimiser.step()

                loss_sum += float(loss.item()) * xb.size(0)
                correct += int((out.argmax(1) == yb).sum().item())
                total += int(xb.size(0))
            scheduler.step()

            entry = {
                "epoch": epoch,
                "train_loss": loss_sum / max(total, 1),
                "train_acc": correct / max(total, 1),
            }

            if val_dl is not None:
                vp, vt = self._infer(val_dl, with_labels=True)
                pred = vp.argmax(1)
                entry["val_acc"] = float((pred == vt).mean())
                # Balanced accuracy, not plain accuracy, is the selection criterion. Plain
                # accuracy would let the network drop the glioma class entirely and still
                # look like it improved - which is exactly the failure we are here to fix.
                recalls = [float((pred[vt == c] == c).mean()) for c in range(self.n_classes)
                           if (vt == c).any()]
                entry["val_balanced_acc"] = float(np.mean(recalls)) if recalls else 0.0

                if entry["val_balanced_acc"] > best_score:
                    best_score = entry["val_balanced_acc"]
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self.net.state_dict().items()}

            self.history.append(entry)
            if self.verbose:
                msg = (f"      epoch {epoch:>2}/{self.epochs}  "
                       f"loss {entry['train_loss']:.4f}  acc {entry['train_acc']*100:5.2f}%")
                if "val_balanced_acc" in entry:
                    msg += (f"   val acc {entry['val_acc']*100:5.2f}%  "
                            f"val bal.acc {entry['val_balanced_acc']*100:5.2f}%")
                print(msg)

        # Restore the epoch that generalised best rather than simply the last one. Without
        # this we would ship whatever the final epoch happened to land on, which on a small
        # noisy dataset is often past the point where the model started memorising.
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_val_balanced_acc = best_score
            if self.verbose:
                print(f"      restored best epoch (val balanced acc {best_score*100:.2f}%)")

        self.net.eval()
        return self

    # ---------------------------------------------------------------- inference
    def _infer(self, loader, with_labels: bool):
        assert self.net is not None
        self.net.eval()
        probs, targets = [], []
        with torch.no_grad():
            for batch in loader:
                if with_labels:
                    xb, yb = batch
                    targets.append(np.asarray(yb))
                else:
                    xb = batch
                xb = xb.to(self.device)
                out = self.net(xb)
                probs.append(F.softmax(out, dim=1).cpu().numpy())
        p = np.vstack(probs)
        if with_labels:
            return p, np.concatenate(targets)
        return p

    def predict_proba(self, images: np.ndarray, tta: bool | None = None) -> np.ndarray:
        """Class probabilities, optionally averaged over test-time augmentations.

        TEST-TIME AUGMENTATION. We run the image through the network several times under
        small label-preserving transforms and average the softmax outputs. This is not a
        trick to inflate a score - it is the matching half of the training augmentation,
        and it targets the same measured problem.

        The concrete failure it addresses: this dataset's test images arrive at different
        scales and framings from the training images (docs/06_DATASETS.md), so a single
        forward pass samples one arbitrary point in that space and can land badly. Averaging
        over a few nearby points is a variance reduction, and because each transform
        preserves the label - a mirrored glioma is still a glioma, a slightly zoomed
        meningioma is still a meningioma - it cannot bias the answer.

        It costs a linear factor in inference time, which is why it is a flag: the API's
        per-request path can turn it off if latency matters more than the last point of
        accuracy.
        """
        require_torch()
        use_tta = self.tta if tta is None else tta
        if not use_tta:
            ds = _MRIDataset(images, None, train=False, seed=self.seed)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
            return self._infer(dl, with_labels=False)

        import cv2

        views = []
        for flip in (False, True):
            for zoom in (1.0, 0.9):
                batch = images
                if zoom != 1.0:
                    # centre crop then resize back - a mild scale perturbation
                    out = []
                    for im in images:
                        h, w = im.shape[:2]
                        ch, cw = int(h * zoom), int(w * zoom)
                        y0, x0 = (h - ch) // 2, (w - cw) // 2
                        out.append(cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h),
                                              interpolation=cv2.INTER_LINEAR))
                    batch = np.stack(out)
                if flip:
                    batch = batch[:, :, ::-1].copy()
                views.append(batch)

        acc = None
        for v in views:
            ds = _MRIDataset(v, None, train=False, seed=self.seed)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
            p = self._infer(dl, with_labels=False)
            acc = p if acc is None else acc + p
        return acc / len(views)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(1)

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path | str) -> None:
        require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "n_classes": self.n_classes,
                "history": self.history,
                "best_val_balanced_acc": self.best_val_balanced_acc,
                "input_size": CNN_INPUT_SIZE,
                "arch": "resnet18",
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "DeepTumorClassifier":
        require_torch()
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(n_classes=int(blob["n_classes"]), device=device, verbose=False,
                  pretrained=False)
        obj.net = build_backbone(obj.n_classes, obj.seed, pretrained=False).to(obj.device)
        obj.net.load_state_dict(blob["state_dict"])
        obj.net.eval()
        obj.history = blob.get("history", [])
        obj.best_val_balanced_acc = blob.get("best_val_balanced_acc")
        return obj


# ======================================================================================
# ENSEMBLE
# ======================================================================================
def blend_probabilities(classical: np.ndarray, deep: np.ndarray, weight: float) -> np.ndarray:
    """Weighted soft vote. `weight` is the share given to the DEEP model, in [0, 1]."""
    w = float(np.clip(weight, 0.0, 1.0))
    return (1.0 - w) * np.asarray(classical) + w * np.asarray(deep)


def choose_blend_weight(classical: np.ndarray, deep: np.ndarray, y_true: np.ndarray,
                        grid: np.ndarray | None = None) -> dict:
    """Pick the ensemble weight on VALIDATION data, by balanced accuracy.

    This must never see the test set. The whole point of reporting a held-out number is
    that every knob - model choice, threshold, and this weight - was set without it.

    Balanced accuracy is the objective for the same reason it drives epoch selection: it
    refuses to let a weight that quietly sacrifices the glioma class look like an
    improvement because the three easy classes got slightly better.
    """
    from sklearn.metrics import balanced_accuracy_score

    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)

    y_true = np.asarray(y_true)
    scored = []
    for w in grid:
        pred = blend_probabilities(classical, deep, float(w)).argmax(1)
        scored.append((float(w), float(balanced_accuracy_score(y_true, pred))))

    best_w, best_score = max(scored, key=lambda r: (r[1], -abs(r[0] - 0.5)))
    return {
        "weight_deep": best_w,
        "val_balanced_accuracy": best_score,
        "curve": [{"weight_deep": w, "val_balanced_accuracy": s} for w, s in scored],
        "objective": "max balanced accuracy on the held-out validation split",
    }
