# preview_mosaics_v5.py
# pip install rasterio numpy matplotlib
import glob, rasterio, numpy as np, matplotlib.pyplot as plt

PALETTE = {10:(38,115,38),20:(163,255,115),30:(255,255,115),40:(255,170,0),
           50:(197,0,0),60:(210,210,210),70:(255,255,255),80:(0,112,192),
           90:(0,176,240),95:(102,255,204),100:(170,170,255)}

def colorize(arr):
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    for v,c in PALETTE.items(): rgb[arr==v]=c
    return rgb

paths = sorted(glob.glob('data/gee_exports_v5/*_mosaic.tif'))[:6]
for p in paths:
    with rasterio.open(p) as ds:
        rgb = np.stack([ds.read(3), ds.read(2), ds.read(1)], axis=2).astype('float32')/10000.0  # B4,B3,B2
        rgb = np.clip(rgb**(1/2.2),0,1)
        lab = ds.read(7)
    fig,ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(rgb); ax[0].set_title(p.split('/')[-1]); ax[0].axis('off')
    ax[1].imshow(colorize(lab)); ax[1].set_title('Label'); ax[1].axis('off')
    plt.tight_layout(); plt.savefig(p.replace('.tif','_preview.png'), dpi=140); plt.close()
