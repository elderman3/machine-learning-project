# predict_knn_map_v5.py
# pip install rasterio joblib numpy matplotlib
import numpy as np, rasterio
from rasterio.windows import Window
from joblib import load
import matplotlib.pyplot as plt

CLASS_VALUES = [10,20,30,40,50,60,70,80,90,95,100]
IDX2CODE = {i:v for i,v in enumerate(CLASS_VALUES)}
PALETTE = {10:(38,115,38),20:(163,255,115),30:(255,255,115),40:(255,170,0),
           50:(197,0,0),60:(210,210,210),70:(255,255,255),80:(0,112,192),
           90:(0,176,240),95:(102,255,204),100:(170,170,255)}

def predict(mosaic_path, model_path, out_tif, out_png):
    model = load(model_path)
    with rasterio.open(mosaic_path) as ds:
        profile = ds.profile.copy(); profile.update(count=1, dtype='int16')
        H,W = ds.height, ds.width
        with rasterio.open(out_tif, 'w', **profile) as dst:
            bh,bw = 1024,1024
            for r0 in range(0,H,bh):
                for c0 in range(0,W,bw):
                    h=min(bh,H-r0); w=min(bw,W-c0); win=Window(c0,r0,w,h)
                    X = ds.read([1,2,3,4,5,6], window=win).astype('float32')/10000.0
                    pred = model.predict(X.reshape(6,-1).T).astype('int16')
                    codes = np.vectorize(IDX2CODE.get)(pred).reshape(h,w).astype('int16')
                    dst.write(codes, 1, window=win)
    # colorize
    with rasterio.open(out_tif) as ds: lab = ds.read(1)
    rgb = np.zeros((lab.shape[0], lab.shape[1], 3), dtype=np.uint8)
    for v,c in PALETTE.items(): rgb[lab==v]=c
    plt.figure(figsize=(8,8)); plt.imshow(rgb); plt.axis('off'); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# Example:
predict('data/us_kansas_grass_val_mosaic.tif', 'knn_worldcover.joblib', 'pred_knn_val.tif', 'pred_knn_val.png')
