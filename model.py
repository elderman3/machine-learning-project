import ee
import geemap.core as geemap


ee.Authenticate()

id = ""
with open("project_id.txt", "r") as f:
    id = f.readline().rstrip()

ee.Initialize(project=id)

