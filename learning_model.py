# train_knn_and_cnn.py
# KNN on per-pixel vectors. CNN on 15x15 spatial patches rebuilt from lon/lat + block_id.
# CSV columns required: B2,B3,B4,B8,B11,B12,label,block_id,split,lon,lat

import math, argparse, pandas as pd, numpy as np
from collections import defaultdict

# ---------- KNN ----------
def run_knn(csv_path):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    df = pd.read_csv(csv_path)
    bands = ["B2","B3","B4","B8","B11","B12"]
    for b in bands: df[b] = df[b].astype(np.float32) / 10000.0
    y = df["label"].astype(np.int64).values
    X = df[bands].values
    split = df["split"].astype(int).values

    Xtr, ytr = X[split==0], y[split==0]
    Xva, yva = X[split==1], y[split==1]
    Xte, yte = X[split==2], y[split==2]

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(Xtr, ytr)

    for name, X_, y_ in [("val", Xva, yva), ("test", Xte, yte)]:
        p = knn.predict(X_)
        print(f"[KNN] {name} acc={accuracy_score(y_, p):.4f} macroF1={f1_score(y_, p, average='macro'):.4f}")
    print("[KNN] report (test):")
    print(classification_report(yte, knn.predict(Xte), digits=3))

# ---------- CNN ----------
# Build 15x15 patches per pixel using metric projection and block grid indices
def run_cnn(csv_path, patch=15, epochs=15, batch_size=128, lr=1e-3):
    import torch, torch.nn as nn, torch.utils.data as td
    from pyproj import Transformer
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.read_csv(csv_path)
    bands = ["B2","B3","B4","B8","B11","B12"]
    df[bands] = (df[bands].astype(np.float32) / 10000.0)
    df["label"] = df["label"].astype(np.int64)
    df["split"] = df["split"].astype(np.int64)
    df["block_id"] = df["block_id"].astype(np.int64)

    # --- compute per-pixel grid indices inside each 1km tile in EPSG:3857 ---
    transformer = Transformer.from_crs("EPSG:4326","EPSG:3857", always_xy=True)
    x_m, y_m = transformer.transform(df["lon"].values, df["lat"].values)
    bx = np.floor(x_m/1000).astype(np.int64)
    by = np.floor(y_m/1000).astype(np.int64)
    # sanity: matches block_id encoding used during export
    enc = bx*10_000_000 + by
    if not np.all(enc == df["block_id"].values):
        # tolerate small mismatches from boundary rounding
        mism = np.sum(enc != df["block_id"].values)
        print(f"[CNN] Warning: {mism} rows mismatch computed block_id.")

    px = np.floor((x_m - bx*1000)/10).astype(np.int16)  # 0..99
    py = np.floor((y_m - by*1000)/10).astype(np.int16)
    df["_px"] = px; df["_py"] = py

    # --- per-band normalization from train split ---
    m = df[df["split"]==0][bands].mean().values
    s = df[df["split"]==0][bands].std(ddof=0).replace(0,np.finfo("f").eps).values
    df[bands] = (df[bands] - m) / s

    # --- index rows by (block_id, px, py) to fetch neighbors fast ---
    idx_map = {}
    for i,(bid, ix, iy) in enumerate(zip(df["block_id"].values, df["_px"].values, df["_py"].values)):
        idx_map[(int(bid), int(ix), int(iy))] = i

    half = patch//2
    H = W = 100  # 1km at 10m
    def has_full_patch(bid, ix, iy):
        return (ix>=half and iy>=half and ix< W-half and iy< H-half)

    # candidate centers per split that have a full patch present
    centers = {k: [] for k in [0,1,2]}
    for i,row in df.iterrows():
        bid, ix, iy, sp = int(row["block_id"]), int(row["_px"]), int(row["_py"]), int(row["split"])
        if has_full_patch(bid, ix, iy):
            centers[sp].append((bid, ix, iy))

    class PatchDS(td.Dataset):
        def __init__(self, centers):
            self.centers = centers
        def __len__(self): return len(self.centers)
        def __getitem__(self, i):
            bid, cx, cy = self.centers[i]
            xs = []
            for yy in range(cy-half, cy+half+1):
                for xx in range(cx-half, cx+half+1):
                    j = idx_map.get((bid, xx, yy), None)
                    if j is None:  # should not happen due to has_full_patch
                        xs.append(np.zeros(6, dtype=np.float32))
                    else:
                        xs.append(df.iloc[j][bands].values.astype(np.float32))
            x = np.stack(xs,0).reshape(patch, patch, 6).transpose(2,0,1)  # C,H,W
            y = int(df.iloc[idx_map[(bid,cx,cy)]]["label"])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    train_ds = PatchDS(centers[0]); val_ds = PatchDS(centers[1]); test_ds = PatchDS(centers[2])
    print(f"[CNN] patches: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = td.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = td.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = td.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    C = int(df["label"].max()+1)

    class SmallCNN(nn.Module):
        def __init__(self, in_ch=6, n_cls=C):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n_cls))
        def forward(self, x):
            z = self.net(x).flatten(1)
            return self.head(z)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                ps.extend(torch.argmax(logits,1).cpu().numpy())
                ys.extend(yb.cpu().numpy())
        return float(accuracy_score(ys, ps)), float(f1_score(ys, ps, average="macro"))

    best = (-1.0, None)
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        acc, f1m = eval_loader(val_loader)
        if f1m > best[0]: best = (f1m, {k:v.cpu().clone() for k,v in model.state_dict().items()})
        print(f"[CNN] epoch {ep:02d} val acc={acc:.4f} macroF1={f1m:.4f}")

    if best[1] is not None:
        model.load_state_dict(best[1])
    acc, f1m = eval_loader(test_loader)
    print(f"[CNN] test acc={acc:.4f} macroF1={f1m:.4f}")

if __name__ == "__main__":
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patch", type=int, default=15)
    args = ap.parse_args()
    """

    #run_knn("s2_worldcover_2021_FR_100tiles_allpixels-2.csv")
    run_cnn("s2_worldcover_2021_FR_100tiles_allpixels-2.csv")
