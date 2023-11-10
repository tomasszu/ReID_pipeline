from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

class Vehicles(BaseDoc):
  vehicle_id: str = ''
  embedding: NdArray[512]


def add_vehicle(vehicle_id, embedding, intersection):
  # Specify your workspace path
  db = HNSWVectorDB[Vehicles](workspace=f'./vectordb/{intersection}/')

  # Index a list of documents with random embeddings
  feature_list = [Vehicles(vehicle_id=vehicle_id, embedding=embedding)]
  db.index(inputs=DocList[Vehicles](feature_list))



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
  db = HNSWVectorDB[Vehicles](workspace=f'./vectordb/{intersection}/')

  # Perform a search query
  query = Vehicles(text='query', embedding=embedding)
  
  try:
    results = db.search(inputs=DocList[Vehicles]([query]), limit=1)
  except:
    return -1
  
  return results[0].matches
