from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

import misc.lance_db_init as create_db


class Vehicles(BaseDoc):
  vehicle_id: str = ''
  embedding: NdArray[512]


def add_vehicle(vehicle_id, embedding, intersection, db):
  
  tbl = db.open_table(intersection)

  data = [
      {"vehicle_id": vehicle_id, "vector": embedding},
  ]

  tbl.add(data=data)

def update_vehicle(vehicle_id, embedding, intersection, db):
  tbl = db.open_table(intersection)

  df = (tbl.search(np.zeros(512), vector_column_name="vector")
      .where(f"vehicle_id = '{vehicle_id}'", prefilter=True)
      .select(["vector", "vehicle_id"])
      .limit(1)
      .to_list())

  if(df):
    if(df[0]['vehicle_id'] == f'{vehicle_id}'):
      embedding_sum = (embedding + df[0]['vector']) / 2
      tbl.update(where=f"vehicle_id = '{vehicle_id}'", values={"vector": embedding_sum})
      print(f"{vehicle_id}. embedding updated")
    else:
      add_vehicle(vehicle_id, embedding, intersection, db)
      print(f"{vehicle_id}. embedding added")
  else:
    add_vehicle(vehicle_id, embedding, intersection, db)
    print(f"{vehicle_id}. embedding added")
  
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
        .metric("dot") \
        .to_list()
    return df
  except:
    return -1

