import lancedb
import numpy as np
db = lancedb.connect("lancedb")

# # data = [{"vector": np.random.rand(128), "lat": 45.5, "long": -122.7},
# #         {"vector": np.random.rand(128), "lat": 40.1, "long": -74.1}]

# # table = db.create_table("my_table", data, mode="overwrite")

# # db["my_table"].head()

# tbl = db.open_table("my_table")

# # Get the updated table as a pandas DataFrame
# df = tbl.to_pandas()

# # Print the DataFrame
# print(df)

# # df = tbl.search(np.random.rand(128)) \
# #     .limit(2) \
# #     .to_list()

# # df = tbl.search(np.zeros(128)) \
# #    .where("""(
# #     (lat IS 45.5)
# #     """)

# #df = tbl.search(where="lat = 45.5")

# df = (tbl.search(np.zeros(128), vector_column_name="vector")
#     .where("lat = 45.5", prefilter=True)
#     .select(["lat", "long"])
#     .limit(2)
#     .to_pandas())

# print(df)

# --------------------------

import pandas as pd
import pyarrow as pa

# schema = pa.schema(
#   [
#       pa.field("vector", pa.list_(pa.float32(), 128)),
#       pa.field("item", pa.string()),
#       pa.field("price", pa.float32()),
#   ])

# tbl = db.create_table("table5", schema=schema, mode="overwrite")

# data = [
#     {"vector": np.random.rand(128), "item": "foo", "price": 10.0},
#     {"vector": np.random.rand(128), "item": "bar", "price": 20.0},
# ]

# tbl.add(data=data)

tbl = db.open_table("table5")

# Get the updated table as a pandas DataFrame
df = tbl.to_pandas()

# Print the DataFrame
print(df)

df = (tbl.search(np.zeros(128), vector_column_name="vector")
    .where("item = 'foo'", prefilter=True)
    .select(["item", "price"])
    .limit(2)
    .to_pandas())

print(df)











