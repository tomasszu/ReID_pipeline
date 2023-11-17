import numpy as np
import lancedb
import pyarrow as pa

import os
import sys

SCHEMA = pa.schema(
  [
      pa.field("vehicle_id", pa.string()),
      pa.field("vector", pa.list_(pa.float32(), 512)),
  ])

def _init_(folder):
  db = lancedb.connect("lancedb")

  try:
    db.create_table(folder, data=None, schema=SCHEMA)
    (f"{folder} table created")
    return db
  except:
    print(f"{folder} table exists")
    return db
