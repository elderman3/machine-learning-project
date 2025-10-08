# build_dataset_v6.py
# pip install numpy rasterio tqdm matplotlib
import os, glob, numpy as np, rasterio
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = 'data'
OUT = 'dataset_worldcover_v6.npz'
CLASS_VALUES = [10,20,30,40,50,60,70,80,90,95,100]
CLASS_SET = set(CLASS_VALUES)
CLASS_MAP = {v:i for i,v in enumerate(CLASS_VALUES)}

# Targets
KNN_TRAIN_PER_CLASS = 20000
KNN_VAL_PER_CLASS   = 3000
KNN_TEST_PER_CLASS  = 3000
PATCH = 64
STRIDE = 64
CNN_TRAIN_PER_CLASS = 400
CNN_VAL_PER_CLASS   = 100
MAJ_THRESH = 0.6

np.random.seed(42)

def split_files():
    allf = sorted(glob.glob(os.path.join(DATA_DIR, '*_mosaic*.tif')))
    tr = [f for f in allf if '_train_' in os.path.basename(f)]
    va = [f for f in allf if '_val_'   in os.path.basename(f)]
    te = [f for f in allf if '_test_'  in os.path.basename(f)]
    if not allf:
        raise FileNotFoundError(f'No files matching {DATA_DIR}/*_mosaic*.tif')
    print(f'found files train/val/test: {len(tr)}/{len(va)}/{len(te)}')
    return tr, va, te

def detect_label_band(ds):
    # Prefer named band
    if ds.descriptions:
        names = [n or '' for n in ds.descriptions]
        if 'label' in names:
            return names.index('label') + 1
    # Fallback: find band whose values are mostly valid ESA codes or 0
    valid = np.array(sorted(list(CLASS_SET | {0})), dtype=ds.dtypes[0])
    best_b, best_score = 1, -1.0
    for b in range(1, ds.count+1):
        arr = ds.read(b, window=Window(0,0, min(2048, ds.width), min(2048, ds.height)))
        u = np.unique(arr)
        score = np.isin(u, valid).mean()
        if score > best_score:
            best_b, best_score = b, score
    return best_b

def detect_band_indices(ds):
    # Expect 6 feature bands. Try by names; else assume first 6.
    if ds.descriptions:
        name_to_idx = {n:i+1 for i,n in enumerate(ds.descriptions) if n}
        wanted = ['B2','B3','B4','B8','B11','B12']
        if all(w in name_to_idx for w in wanted):
            return [name_to_idx[w] for w in wanted]
    return [1,2,3,4,5,6]

def collect_knn(paths, per_class):
    X, y = [], []
    got = defaultdict(int)
    for p in paths:
        with rasterio.open(p) as ds:
            feat_idx = detect_band_indices(ds)
            lab_b = detect_label_band(ds)
            H, W = ds.height, ds.width
            bh, bw = 1024, 1024
            for r0 in range(0, H, bh):
                for c0 in range(0, W, bw):
                    h = min(bh, H-r0); w = min(bw, W-c0)
                    win = Window(c0, r0, w, h)
                    lab = ds.read(lab_b, window=win)
                    for code, cls in CLASS_MAP.items():
                        need = per_class - got[cls]
                        if need <= 0: continue
                        m = (lab == code)
                        if not m.any(): continue
                        idx = np.argwhere(m)
                        if idx.shape[0] > need:
                            idx = idx[np.random.choice(idx.shape[0], need, replace=False)]
                        bands = ds.read(feat_idx, window=win).astype('float32') / 10000.0
                        feats = bands[:, idx[:,0], idx[:,1]].T
                        X.append(feats); y.append(np.full(len(idx), cls, 'int64'))
                        got[cls] += len(idx)
    X = np.concatenate(X, axis=0) if X else np.zeros((0,6), 'float32')
    y = np.concatenate(y, axis=0) if y else np.zeros((0,), 'int64')
    return X, y

def collect_patches(paths, per_class):
    X, y = [], []
    got = defaultdict(int)
    for p in paths:
        with rasterio.open(p) as ds:
            feat_idx = detect_band_indices(ds)
            lab_b = detect_label_band(ds)
            H, W = ds.height, ds.width
            for r0 in range(0, H-PATCH+1, STRIDE):
                for c0 in range(0, W-PATCH+1, STRIDE):
                    if all(got[k] >= per_class for k in CLASS_MAP.values()): break
                    win = Window(c0, r0, PATCH, PATCH)
                    lab = ds.read(lab_b, window=win)
                    vals, cnts = np.unique(lab, return_counts=True)
                    dom_i = np.argmax(cnts); dom_val = int(vals[dom_i]); frac = cnts[dom_i]/(PATCH*PATCH)
                    if dom_val not in CLASS_MAP or frac < MAJ_THRESH: continue
                    cls = CLASS_MAP[dom_val]
                    if got[cls] >= per_class: continue
                    bands = ds.read(feat_idx, window=win).astype('float32') / 10000.0
                    X.append(bands); y.append(cls); got[cls] += 1
    X = np.stack(X, axis=0) if X else np.zeros((0,6,PATCH,PATCH), 'float32')
    y = np.array(y, 'int64')
    return X, y

def report(name, y):
    cnt = np.bincount(y, minlength=len(CLASS_VALUES))
    missing = [int(CLASS_VALUES[i]) for i,c in enumerate(cnt) if c==0]
    print(f'{name} counts:', dict(zip(map(int,CLASS_VALUES), cnt.tolist())), 'missing:', missing)
    plt.figure(figsize=(7,3))
    plt.bar(range(len(CLASS_VALUES)), cnt)
    plt.xticks(range(len(CLASS_VALUES)), list(map(str,CLASS_VALUES)), rotation=45, ha='right')
    plt.title(name); plt.tight_layout(); plt.savefig(f'{name.replace(" ","_").lower()}_hist.png', dpi=140); plt.close()

train_paths, val_paths, test_paths = split_files()

Xtr_knn,ytr_knn = collect_knn(train_paths, KNN_TRAIN_PER_CLASS); report('KNN train', ytr_knn)
Xva_knn,yva_knn = collect_knn(val_paths,   KNN_VAL_PER_CLASS);   report('KNN val',   yva_knn)
Xte_knn,yte_knn = collect_knn(test_paths,  KNN_TEST_PER_CLASS);  report('KNN test',  yte_knn)

Xtr_cnn,ytr_cnn = collect_patches(train_paths, CNN_TRAIN_PER_CLASS); report('CNN train', ytr_cnn)
Xva_cnn,yva_cnn = collect_patches(val_paths,   CNN_VAL_PER_CLASS);   report('CNN val',   yva_cnn)

np.savez_compressed(
    OUT,
    CLASS_VALUES=np.array(CLASS_VALUES, 'int16'),
    Xtr_knn=Xtr_knn, ytr_knn=ytr_knn,
    Xva_knn=Xva_knn, yva_knn=yva_knn,
    Xte_knn=Xte_knn, yte_knn=yte_knn,
    Xtr_cnn=Xtr_cnn, ytr_cnn=ytr_cnn,
    Xva_cnn=Xva_cnn, yva_cnn=yva_cnn,
    PATCH=np.array([PATCH], 'int16')
)
print('Saved', OUT)
