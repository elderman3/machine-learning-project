# predict_cnn_map_fixed.py
# pip install rasterio torch torchvision numpy matplotlib
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

PALETTE = {10:(38,115,38),20:(163,255,115),30:(255,255,115),40:(255,170,0),
           50:(197,0,0),60:(210,210,210),70:(255,255,255),80:(0,112,192),
           90:(0,176,240),95:(102,255,204),100:(170,170,255)}

class SimpleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6,32,3,padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(256, n)
    def forward(self, x): return self.fc(self.net(x).flatten(1))

def colorize(arr):
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), np.uint8)
    for v,c in PALETTE.items(): rgb[arr==v]=c
    return rgb

@torch.no_grad()
def predict_cnn_map(mosaic_path, weights_path, dataset_npz, out_tif, out_png,
                    stride=16, tile=1024, device=None, batch=512):
    d = np.load(dataset_npz, allow_pickle=True)
    CLASS_VALUES = d['CLASS_VALUES']; K = len(CLASS_VALUES)
    IDX2CODE = {i:int(CLASS_VALUES[i]) for i in range(K)}
    PATCH = int(d['PATCH'][0])
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN(K).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    pad_top = PATCH//2; pad_left = PATCH//2
    pad_bot = PATCH - 1 - PATCH//2; pad_right = PATCH - 1 - PATCH//2

    with rasterio.open(mosaic_path) as ds:
        H, W = ds.height, ds.width
        prof = ds.profile.copy(); prof.update(count=1, dtype='int16')
        with rasterio.open(out_tif, 'w', **prof) as dst:
            for r0 in range(0, H, tile):
                for c0 in range(0, W, tile):
                    h = min(tile, H - r0); w = min(tile, W - c0)
                    r1 = max(0, r0 - pad_top); c1 = max(0, c0 - pad_left)
                    r2 = min(H, r0 + h + pad_bot); c2 = min(W, c0 + w + pad_right)
                    big = ds.read([1,2,3,4,5,6], window=Window(c1, r1, c2-c1, r2-r1)).astype('float32')/10000.0

                    Hb, Wb = big.shape[1], big.shape[2]
                    rows = list(range(0, Hb - PATCH + 1, stride))
                    cols = list(range(0, Wb - PATCH + 1, stride))
                    # keep centers inside the interior tile
                    r_off, c_off = r0 - r1, c0 - c1
                    rows_i = [ri for ri in rows if (ri + PATCH//2) >= r_off and (ri + PATCH//2) < (r_off + h)]
                    cols_i = [ci for ci in cols if (ci + PATCH//2) >= c_off and (ci + PATCH//2) < (c_off + w)]
                    if not rows_i or not cols_i:
                        continue

                    grid_probs = np.zeros((K, len(rows_i), len(cols_i)), dtype=np.float32)

                    # batch patches for speed
                    batch_patches, batch_pos = [], []
                    for i, ri in enumerate(rows_i):
                        for j, ci in enumerate(cols_i):
                            patch = big[:, ri:ri+PATCH, ci:ci+PATCH]
                            batch_patches.append(patch); batch_pos.append((i,j))
                            if len(batch_patches) == batch:
                                xb = torch.from_numpy(np.stack(batch_patches)).to(device)
                                probs = torch.softmax(model(xb), dim=1).cpu().numpy()  # [B,K]
                                for (ii,jj), pr in zip(batch_pos, probs): grid_probs[:, ii, jj] = pr
                                batch_patches, batch_pos = [], []
                    if batch_patches:
                        xb = torch.from_numpy(np.stack(batch_patches)).to(device)
                        probs = torch.softmax(model(xb), dim=1).cpu().numpy()
                        for (ii,jj), pr in zip(batch_pos, probs): grid_probs[:, ii, jj] = pr

                    # upsample coarse prob grid to pixel grid
                    t = torch.from_numpy(grid_probs)[None]  # [1,K,nr,nc]
                    up = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0].numpy()  # [K,h,w]
                    lab_idx = up.argmax(0)
                    lab_codes = np.vectorize(IDX2CODE.get)(lab_idx).astype('int16')
                    dst.write(lab_codes, 1, window=Window(c0, r0, w, h))

    with rasterio.open(out_tif) as ds:
        arr = ds.read(1)
    plt.figure(figsize=(8,8)); plt.imshow(colorize(arr)); plt.axis('off'); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


# Example:
#predict_cnn_map('data/shanghai_built_train_mosaic.tif',
#                'cnn_worldcover.pt', 'dataset_worldcover_v6.npz',
#                'pred_cnn_shanghai.tif', 'pred_cnn_shanghai.png',
#                stride=16, tile=1024, batch=512)
predict_cnn_map('data/us_kansas_grass_train_mosaic.tif',
                'cnn_worldcover.pt', 'dataset_worldcover_v6.npz',
                'pred_cnn_us.tif', 'pred_cnn_us.png',
                stride=2, tile=16, batch=256)