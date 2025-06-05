import lancedb
import pyarrow as pa

def _init_(folder, features_size=256):
    """Initialize LanceDB with a dynamically defined feature size."""
    
    SCHEMA = pa.schema(
        [
            pa.field("vehicle_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), features_size)),  # Use dynamic features_size
            pa.field("times_summed", pa.int8()),
        ]
    )

    db = lancedb.connect("lancedb")

    try:
        db.create_table(folder, data=None, schema=SCHEMA)
        print(f"{folder} table created")
        return db
    except:
        #print(f"{folder} table exists")
        return db