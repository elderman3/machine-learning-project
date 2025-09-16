import ee, time
ee.Authenticate()

id = ""
with open("project_id.txt", "r") as f:
    id = f.readline().rstrip()
ee.Initialize(project=id)

# Params
YEAR = 2021 # eyear wher WorldCover is available
aoi = ee.Geometry.Rectangle([0.065, 47.91, 0.335, 48.09])  # latitude, longitude rectangle of interest - rectangle around Le Mans - France


classes_esa = [10,20,30,40,50,60,70,80,90,95,100]
code_to_idx = {c:i for i,c in enumerate(classes_esa)}

# Input data
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(aoi)
      .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)))

# Cloud mask
def mask_s2(img):
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1<<10).Or(qa.bitwiseAnd(1<<11))
    return img.updateMask(cloud.Not())

s2c = s2.map(mask_s2)

# Month -> Year median
s2c = s2.map(mask_s2)
annual = s2c.median().clip(aoi)

# Features 
bands10 = ["B2","B3","B4","B8"] # 10 m
bands20 = ["B11","B12"] # 20 m
feat10 = annual.select(bands10)
feat20 = (annual.select(bands20)
                 .resample('bilinear')
                 .reproject(crs=feat10.projection())) # to 10m resolution
features = feat10.addBands(feat20)

# Only selected bands
features = features.select(bands10 + bands20)

# Labels
wc = ee.Image("ESA/WorldCover/v200/2021").select("Map")
wc = wc.updateMask(wc.remap(classes_esa, [1]*len(classes_esa))) # keep known classes

# ESA codes to {0, 1, 2,... 10}
def remap_codes(img):
    from_c = ee.List(classes_esa)
    to_c = ee.List([code_to_idx[c] for c in classes_esa])
    return img.remap(from_c, to_c)
y = remap_codes(wc).rename("label")

stack = features.addBands(y).clip(aoi)

# Sampling
sample = stack.stratifiedSample(
    numPoints=10000,
    classBand="label",
    region=aoi,
    scale=10,
    geometries=True,
    classValues=list(code_to_idx.values()),
    classPoints=[200]*len(code_to_idx) # balanced
)

# File To Drive
task = ee.batch.Export.table.toDrive(
    collection=sample,
    description="s2_worldcover_samples",
    fileFormat="CSV"
)
task.start()
print("Export Start")

while task.active():
    print(task.status())
    time.sleep(5)
