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

  # !!!!! ----------------------->>>>>>>>>>>> Šiten man vajag table dabuut no db.

  # Perform a search query
  query = Vehicles(text='query', embedding=embedding)
  alt_query = Vehicles(text='query', vehicle_id='1')
  
  try:
    results = db.search(inputs=DocList[Vehicles]([query]), limit=1)
    alt_results = db.search(inputs=DocList[Vehicles]([alt_query]), limit=1)
    print(alt_results)
  except:
    return -1
  
  return results[0].matches
