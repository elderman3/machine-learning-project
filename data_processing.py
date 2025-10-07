# Export ALL 10 m pixels from random 1 km tiles across mainland France as one CSV.
# Columns: B2,B3,B4,B8,B11,B12,label,block_id,split,lon,lat,class_name,.geo

import ee, time
ee.Authenticate()
ee.Initialize(project=open("project_id.txt").read().strip())

# ---------- Params ----------
YEAR = 2021
N_BLOCKS = 100
BLOCK_M = 1000
SEED = 42

classes_esa = [10,20,30,40,50,60,70,80,90,95,100]
code_to_idx = {c:i for i,c in enumerate(classes_esa)}
class_names = ['Tree','Shrub','Grass','Crop','Built','Bare','SnowIce','Water','Wetland','Mangrove','MossLichen']

# ---------- AOI: mainland France ----------
fr = (ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
      .filter(ee.Filter.eq('country_na', 'France'))
      .geometry())
mainland_bbox = ee.Geometry.Rectangle([-5.5, 41.0, 9.8, 51.5])
aoi = fr.intersection(mainland_bbox, 1)

# ---------- Sentinel-2 annual composite ----------
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi)
      .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))

def mask_s2(img):
    qa = img.select("QA60")
    mask = qa.bitwiseAnd(1<<10).eq(0).And(qa.bitwiseAnd(1<<11).eq(0))
    return img.updateMask(mask)

annual = s2.map(mask_s2).median().clip(aoi)

bands10 = ["B2","B3","B4","B8"]
bands20 = ["B11","B12"]
feat10 = annual.select(bands10)
proj10 = feat10.projection()

feat20 = (annual.select(bands20)
          .resample('bilinear')
          .reproject(crs=proj10))

features = feat10.addBands(feat20).select(bands10 + bands20)

# ---------- Labels (ESA WorldCover 2021) ----------
wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
wc = wc.updateMask(wc.remap(classes_esa, [1]*len(classes_esa)))

def remap_codes(img):
    from_c = ee.List(classes_esa)
    to_c = ee.List([code_to_idx[c] for c in classes_esa])
    return img.remap(from_c, to_c)

label = remap_codes(wc).rename("label")

# ---------- Stack and projection ----------
stack = features.addBands(label).clip(aoi).reproject(crs=proj10, scale=10)

# ---------- 1 km grid and block_id in EPSG:3857 ----------
metric_proj = ee.Projection('EPSG:3857').atScale(1)  # meters
coords_m = ee.Image.pixelCoordinates(metric_proj)
bx = coords_m.select('x').divide(BLOCK_M).floor().toInt64()
by = coords_m.select('y').divide(BLOCK_M).floor().toInt64()
block_id = bx.multiply(ee.Number(10_000_000)).add(by).rename('block_id')

# Split from block_id.mod(20): 0=train, 1=val, 2=test
mod20 = block_id.mod(20)
split_img = (ee.Image(0)
             .where(mod20.gte(14), 1)
             .where(mod20.gte(17), 2)
             .rename('split'))

# ---------- select N_BLOCKS random tiles ----------
rand_pts = ee.FeatureCollection.randomPoints(region=aoi, points=N_BLOCKS*20, seed=SEED, maxError=1000)

def attach_block_id(feat):
    pt = feat.geometry()
    bid = block_id.sample(pt, scale=BLOCK_M, projection=metric_proj, numPixels=1).first().get('block_id')
    return feat.set({'block_id': bid})

rand_with_ids = rand_pts.map(attach_block_id).filter(ee.Filter.notNull(['block_id']))
distinct_blocks = rand_with_ids.distinct(['block_id']).randomColumn('rand', seed=SEED).sort('rand').limit(N_BLOCKS)
block_ids_list = ee.List(distinct_blocks.aggregate_array('block_id'))

# Mask to selected blocks only
selected_mask = block_id.remap(block_ids_list, ee.List.repeat(1, block_ids_list.length()), 0).rename('selected')

# ---------- final grid to export ----------
grid = (stack
        .addBands(block_id)
        .addBands(split_img)
        .updateMask(selected_mask))

# ---------- add lon/lat and class_name ----------
def add_ll_and_name(f):
    ll = f.geometry().coordinates()
    lbl = ee.Number(f.get('label')).int()
    return f.set({
        'lon': ll.get(0),
        'lat': ll.get(1),
        'class_name': ee.List(class_names).get(lbl)
    })

# Export ALL unmasked 10 m pixels inside the selected tiles
all_pixels = (grid
    .updateMask(label.mask())  # keep only labeled pixels
    .sample(
        region=aoi,
        scale=10,
        numPixels=1_000_000_000,  # force effectively "all"
        geometries=True,
        tileScale=8               # raise if memory errors
    )
    .map(add_ll_and_name)
)

ordered = all_pixels.select(
    ['B2','B3','B4','B8','B11','B12','label','class_name','block_id','split','lon','lat','.geo']
)

task = ee.batch.Export.table.toDrive(
    collection=ordered,
    description=f"s2_worldcover_{YEAR}_FR_{N_BLOCKS}tiles_allpixels",
    fileFormat="CSV"
)
task.start()
print("Export started")
while task.active():
    print(task.status())
    time.sleep(10)
print("Final:", task.status())
