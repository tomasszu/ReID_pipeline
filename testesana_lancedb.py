import lancedb
db = lancedb.connect("./.lancedb")


data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
        {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1}]

db.create_table("my_table", data)

db["my_table"].head()
