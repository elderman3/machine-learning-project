import os, glob, rasterio, numpy as np
DIR = 'data'
files = sorted(glob.glob(os.path.join(DIR, '*_mosaic*.tif')))
print('total mosaics:', len(files))
print('train/val/test:',
      len([f for f in files if '_train_' in os.path.basename(f)]),
      len([f for f in files if '_val_'   in os.path.basename(f)]),
      len([f for f in files if '_test_'  in os.path.basename(f)]))

for p in files[:3]:
    with rasterio.open(p) as ds:
        print(os.path.basename(p), 'bands:', ds.count, 'names:', ds.descriptions)
        v = np.unique(ds.read(1)[::200,::200])  # quick peek
        print('B1 sample uniq:', v[:10], '...')
        v7 = np.unique(ds.read(ds.count)[::200,::200])
        print(f'last band sample uniq size:{v7.size}')