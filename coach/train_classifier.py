"""Train the exercise classifier (squat / push-up / biceps curl / shoulder press) on 2-second pose windows.

Split by VIDEO: the test set is 10 real videos per exercise that no model ever sees in training.
Training uses all 400 synthetic videos + 12 real videos per exercise; 3 more real videos per exercise
pick the epoch (validation).

Compares three models, then exports the small MLP to docs/model.json for the browser demo:
  1. logistic regression on window statistics      (classic ML baseline)
  2. MLP on window statistics                      (tiny, runs in plain JavaScript)
  3. 1D-CNN over the raw window                    (temporal deep learning)

    python -m coach.train_classifier
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from coach.features import N_FEATURES, sequence_features, window_stats

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["bicep_curl", "push_up", "shoulder_press", "squat"]
WINDOW, STRIDE = 20, 5  # 2 s at 10 fps, a new window every 0.5 s
SEED = 7


def load_videos() -> list[dict]:
    vids = []
    for f in sorted((ROOT / "data" / "landmarks" / "exercises").glob("*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        feats = sequence_features(z["world"])
        if len(feats) >= WINDOW:
            vids.append({"id": f.stem, "label": CLASSES.index(meta["label"]), "source": meta["source"], "feats": feats})
    return vids


def split(vids: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    out = {"train": [v for v in vids if v["source"] == "synthetic"], "val": [], "test": []}
    for c in range(len(CLASSES)):
        real = sorted((v for v in vids if v["source"] == "real" and v["label"] == c), key=lambda v: v["id"])
        rng.shuffle(real)
        out["test"] += real[:10]
        out["val"] += real[10:13]
        out["train"] += real[13:]
    return out


def windows(vids: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, owner = [], [], []
    for i, v in enumerate(vids):
        f = v["feats"]
        for s in range(0, len(f) - WINDOW + 1, STRIDE):
            xs.append(f[s:s + WINDOW]); ys.append(v["label"]); owner.append(i)
    return np.stack(xs), np.array(ys), np.array(owner)


class MLP(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, d_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d_in, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_out))

    def forward(self, x):  # x: (B, T, F) → conv over time
        return self.net(x.transpose(1, 2))


def fit(model: nn.Module, x_tr, y_tr, x_val, y_val, epochs: int = 40) -> nn.Module:
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    xt, yt = torch.tensor(x_tr), torch.tensor(y_tr)
    best = (-1, None)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt)).split(128):
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        acc = accuracy_score(y_val, predict(model, x_val))
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model


@torch.inference_mode()
def predict(model: nn.Module, x) -> np.ndarray:
    model.eval()
    return model(torch.tensor(x)).argmax(1).numpy()


def video_vote(pred: np.ndarray, owner: np.ndarray, vids: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y, p = [], []
    for i, v in enumerate(vids):
        votes = pred[owner == i]
        if len(votes):
            y.append(v["label"]); p.append(np.bincount(votes, minlength=len(CLASSES)).argmax())
    return np.array(y), np.array(p)


def report(name, pred, y, owner, vids) -> dict:
    vy, vp = video_vote(pred, owner, vids)
    r = {"window_accuracy": round(accuracy_score(y, pred), 4), "window_macro_f1": round(f1_score(y, pred, average="macro"), 4),
         "video_accuracy": round(accuracy_score(vy, vp), 4), "videos": int(len(vy)),
         "confusion_videos": confusion_matrix(vy, vp, labels=range(len(CLASSES))).tolist()}
    print(name, {k: v for k, v in r.items() if k != "confusion_videos"}, flush=True)
    return r


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data = split(load_videos())
    print({k: len(v) for k, v in data.items()}, flush=True)
    raw = {k: windows(v) for k, v in data.items()}
    stats = {k: np.stack([window_stats(w) for w in raw[k][0]]) for k in raw}
    mean, std = stats["train"].mean(0), stats["train"].std(0) + 1e-6
    norm = {k: ((v - mean) / std).astype(np.float32) for k, v in stats.items()}
    y = {k: raw[k][1] for k in raw}
    results = {"split": {k: len(v) for k, v in data.items()}, "window_seconds": WINDOW / 10}

    lr = LogisticRegression(C=1.0, max_iter=3000).fit(norm["train"], y["train"])
    results["logistic_regression"] = report("logreg", lr.predict(norm["test"]), y["test"], raw["test"][2], data["test"])

    mlp = fit(MLP(norm["train"].shape[1], len(CLASSES)), norm["train"], y["train"], norm["val"], y["val"])
    results["mlp"] = report("mlp", predict(mlp, norm["test"]), y["test"], raw["test"][2], data["test"])

    fmean, fstd = raw["train"][0].reshape(-1, N_FEATURES).mean(0), raw["train"][0].reshape(-1, N_FEATURES).std(0) + 1e-6
    seq = {k: ((raw[k][0] - fmean) / fstd).astype(np.float32) for k in raw}
    cnn = fit(CNN(N_FEATURES, len(CLASSES)), seq["train"], y["train"], seq["val"], y["val"])
    results["cnn"] = report("cnn", predict(cnn, seq["test"]), y["test"], raw["test"][2], data["test"])

    # domain gap: train on synthetic avatars only, test on the same real videos
    syn = np.array([data["train"][i]["source"] == "synthetic" for i in raw["train"][2]])
    mlp_syn = fit(MLP(norm["train"].shape[1], len(CLASSES)), norm["train"][syn], y["train"][syn], norm["val"], y["val"])
    results["mlp_synthetic_only"] = report("mlp synthetic-only", predict(mlp_syn, norm["test"]), y["test"], raw["test"][2], data["test"])

    # export the MLP for the browser: plain JSON weights + normalisation stats
    sd = {k: v.numpy().round(5).tolist() for k, v in mlp.state_dict().items()}
    model_json = {"classes": CLASSES, "window": WINDOW, "fps": 10, "mean": mean.round(5).tolist(), "std": std.round(5).tolist(),
                  "w1": sd["net.0.weight"], "b1": sd["net.0.bias"], "w2": sd["net.3.weight"], "b2": sd["net.3.bias"]}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "model.json").write_text(json.dumps(model_json, separators=(",", ":")), encoding="utf-8")
    (ROOT / "models").mkdir(exist_ok=True)
    torch.save({"mlp": mlp.state_dict(), "cnn": cnn.state_dict(), "mean": mean, "std": std, "fmean": fmean, "fstd": fstd,
                "classes": CLASSES}, ROOT / "models" / "classifier.pt")
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "classifier.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
