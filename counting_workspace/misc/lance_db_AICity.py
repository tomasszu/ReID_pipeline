from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
import numpy as np
import pyarrow as pa


import counting_workspace.misc.lance_db_init as create_db

def add_vehicle(vehicle_id, embedding, intersection, db):
  
  SCHEMA = pa.schema(
    [
        pa.field("vehicle_id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 512)),
        pa.field("times_summed", pa.int8()),
    ])

  try:
      # Mēģiniet atvērt tabulu
      tbl = db.open_table(intersection)
  except FileNotFoundError:
      # Ja tabula neeksistē, izveidojiet jaunu
      print(f"Tabula {intersection} neeksistē. Izveidojam jaunu tabulu...")
      db.create_table(str(intersection), data=None, schema=SCHEMA)
      tbl = db.open_table(intersection)

  # Atveriet tabulu, jo tā jau eksistē vai tikko izveidota
  tbl = db.open_table(intersection)

  data = [
      {"vehicle_id": vehicle_id, "vector": embedding, "times_summed": 0},
  ]

  tbl.add(data=data)

def update_vehicle(vehicle_id, embedding, intersection, db):
  tbl = db.open_table(intersection)

  df = (tbl.search(np.zeros(512, dtype= np.float16), vector_column_name="vector")
      .where(f"vehicle_id = '{vehicle_id}'", prefilter=True)
      .select(["vector", "vehicle_id", "times_summed"])
      .limit(1)
      .to_list())
  

  if(df):
    if(df[0]['vehicle_id'] == f'{vehicle_id}'):
      times_summed = df[0]['times_summed'] + 1
      #print("!!!Times summed: ", times_summed)
      if(times_summed < 127):
        embedding_sum = (np.array(df[0]['vector']) + (embedding / times_summed)) / (1 + (1 / times_summed))
        #embedding_sum = [round(x, 7) for x in embedding_sum]
        tbl.update(where=f"vehicle_id = '{vehicle_id}'", values={"vector": embedding_sum, "times_summed": times_summed})
        # print(f"{vehicle_id}. embedding updated")
        # print(f"1 + {1/times_summed} / {1 + (1 / times_summed)}")
    else:
      add_vehicle(vehicle_id, embedding, intersection, db)
      # print(f"{vehicle_id}. embedding added")
  else:
    add_vehicle(vehicle_id, embedding, intersection, db)
    # print(f"{vehicle_id}. embedding added")
  
  # print(df)



def query(embedding, intersection):
  db = HNSWVectorDB[Vehicles](workspace=f'./vectordb/{intersection}/')

  # Perform a search query
  query = Vehicles(text='query', embedding=embedding)
  results = db.search(inputs=DocList[Vehicles]([query]), limit=5)
  print(len(results[0].matches))
  # Print out the matches
  for m in results[0].matches:
    print(m)

def query_for_ID(embedding, intersection):
  db = create_db._init_(intersection)

  # Perform a search query
  tbl = db.open_table(intersection)

  try:
    df = tbl.search(embedding) \
        .limit(1) \
        .metric("l2") \
        .to_list()
    return df
  except:
    return -1

