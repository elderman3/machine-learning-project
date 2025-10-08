# export_gee_fixed.py
import ee
ee.Authenticate()
ee.Initialize(project=open("project_id.txt").read().strip())

YEAR = ('2021-01-01', '2021-12-31')
CLOUD_PROB_THRESH = 40
BANDS = ['B2','B3','B4','B8','B11','B12']
DRIVE_FOLDER = 'gee_exports_v5'
AOIS = [
    ('amazon_manaus_train', ee.Geometry.Rectangle([-60.984952, -3.194916, -60.894984, -3.105084])),
    ('amazon_manaus_val',   ee.Geometry.Rectangle([-60.894984, -3.194916, -60.805016, -3.105084])),
    ('amazon_manaus_test',  ee.Geometry.Rectangle([-60.805016, -3.194916, -60.715048, -3.105084])),

    ('sahara_mauritania_train', ee.Geometry.Rectangle([-9.394825, 21.455084, -9.298275, 21.544916])),
    ('sahara_mauritania_val',   ee.Geometry.Rectangle([-9.298275, 21.455084, -9.201725, 21.544916])),
    ('sahara_mauritania_test',  ee.Geometry.Rectangle([-9.201725, 21.455084, -9.105175, 21.544916])),

    ('australia_shrub_train', ee.Geometry.Rectangle([121.347390, -28.044916, 121.449130, -27.955084])),
    ('australia_shrub_val',   ee.Geometry.Rectangle([121.449130, -28.044916, 121.550870, -27.955084])),
    ('australia_shrub_test',  ee.Geometry.Rectangle([121.550870, -28.044916, 121.652610, -27.955084])),

    ('punjab_india_crop_train', ee.Geometry.Rectangle([75.393127, 30.755084, 75.497709, 30.844916])),
    ('punjab_india_crop_val',   ee.Geometry.Rectangle([75.497709, 30.755084, 75.602291, 30.844916])),
    ('punjab_india_crop_test',  ee.Geometry.Rectangle([75.602291, 30.755084, 75.706873, 30.844916])),

    ('sundarbans_mangrove_train', ee.Geometry.Rectangle([88.654671, 21.955084, 88.751557, 22.044916])),
    ('sundarbans_mangrove_val',   ee.Geometry.Rectangle([88.751557, 21.955084, 88.848443, 22.044916])),
    ('sundarbans_mangrove_test',  ee.Geometry.Rectangle([88.848443, 21.955084, 88.945329, 22.044916])),

    ('us_kansas_grass_train', ee.Geometry.Rectangle([-100.372899, 38.755084, -100.257633, 38.844916])),
    ('us_kansas_grass_val',   ee.Geometry.Rectangle([-100.257633, 38.755084, -100.142367, 38.844916])),
    ('us_kansas_grass_test',  ee.Geometry.Rectangle([-100.142367, 38.755084, -100.027101, 38.844916])),

    ('shanghai_built_train', ee.Geometry.Rectangle([121.392470, 31.155084, 121.497490, 31.244916])),
    ('shanghai_built_val',   ee.Geometry.Rectangle([121.497490, 31.155084, 121.602510, 31.244916])),
    ('shanghai_built_test',  ee.Geometry.Rectangle([121.602510, 31.155084, 121.707530, 31.244916])),

    ('greenland_ice_train', ee.Geometry.Rectangle([-48.211263, 68.055084, -47.970421, 68.144916])),
    ('greenland_ice_val',   ee.Geometry.Rectangle([-47.970421, 68.055084, -47.729579, 68.144916])),
    ('greenland_ice_test',  ee.Geometry.Rectangle([-47.729579, 68.055084, -47.488737, 68.144916])),

    ('lake_victoria_water_train', ee.Geometry.Rectangle([32.915231, -1.094916, 33.005077, -1.005084])),
    ('lake_victoria_water_val',   ee.Geometry.Rectangle([33.005077, -1.094916, 33.094923, -1.005084])),
    ('lake_victoria_water_test',  ee.Geometry.Rectangle([33.094923, -1.094916, 33.184769, -1.005084])),

    ('sudd_wetland_train', ee.Geometry.Rectangle([30.364091, 7.455084, 30.454697, 7.544916])),
    ('sudd_wetland_val',   ee.Geometry.Rectangle([30.454697, 7.455084, 30.545303, 7.544916])),
    ('sudd_wetland_test',  ee.Geometry.Rectangle([30.545303, 7.455084, 30.635909, 7.544916])),

    ('nunavut_lichen_train', ee.Geometry.Rectangle([-84.853601, 67.555084, -84.617867, 67.644916])),
    ('nunavut_lichen_val',   ee.Geometry.Rectangle([-84.617867, 67.555084, -84.382133, 67.644916])),
    ('nunavut_lichen_test',  ee.Geometry.Rectangle([-84.382133, 67.555084, -84.146399, 67.644916])),
]

s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(*YEAR)
s2cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
wc = ee.Image('ESA/WorldCover/v200/2021').select('Map').toUint16().rename('label')

# join cloud prob
join = ee.Join.saveFirst('cloud_prob')
cond = ee.Filter.equals(leftField='system:index', rightField='system:index')
s2j = ee.ImageCollection(join.apply(primary=s2, secondary=s2cloud, condition=cond))

def mask_clouds(i):
    cp = ee.Image(i.get('cloud_prob')).select('probability')
    return ee.Image(i).updateMask(cp.lt(CLOUD_PROB_THRESH))

def prep(i):
    f32 = ee.Image(i).select(BANDS).toFloat()
    b10 = f32.select(['B2','B3','B4','B8'])
    b20 = f32.select(['B11','B12']).resample('bilinear')
    f32 = b10.addBands(b20).clamp(0, 10000)
    return f32.round().toUint16()  # 0..10000

s2c = s2j.map(mask_clouds).map(prep)

def composite(geom):
    col = s2c.filterBounds(geom)          # s2c is UInt16 per-image
    comp = col.median()                   # -> Float64
    ref  = col.first().select('B2').projection()
    comp = comp.round().toUint16()        # back to UInt16
    return comp.reproject(ref.atScale(10))

for name, geom in AOIS:
    comp = composite(geom)
    lbl   = wc.reproject(comp.projection())
    stack = comp.addBands(wc).clip(geom)  # all UInt16, label is nearest
    ee.batch.Export.image.toDrive(
        image=stack, description=f'{name}_mosaic',
        folder=DRIVE_FOLDER, fileNamePrefix=f'{name}_mosaic',
        region=geom, scale=10, maxPixels=1e13
    ).start()
