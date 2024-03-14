import lancedb
import pyarrow as pa

SCHEMA = pa.schema(
  [
      pa.field("vehicle_id", pa.string()),
      pa.field("vector", pa.list_( pa.float32(), 1024)), # VehicleReID + CLIP
      #pa.field("vector", pa.list_(pa.float32(), 512)), # Priekš VehicleReID
      pa.field("times_summed", pa.int8()),
  ])

def _init_(folder):
  db = lancedb.connect("lancedb")

  try:
    db.create_table(folder, data=None, schema=SCHEMA)
    (f"{folder} table created")
    return db
  except:
    #print(f"{folder} table exists")
    return db
