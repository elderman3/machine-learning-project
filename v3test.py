# quick_label_check.py
import glob, rasterio, numpy as np
CLS = {10,20,30,40,50,60,70,80,90,95,100,0}
for p in glob.glob('data/*_mosaic.tif'):
    with rasterio.open(p) as ds:
        v = ds.read(7)
    uniq = np.unique(v)
    bad = [x for x in uniq if x not in CLS]
    print(p, 'ok' if not bad else f'bad values: {bad}')