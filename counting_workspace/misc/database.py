from docarray import BaseDoc
from docarray.typing import NdArray
from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

class Vehicles(BaseDoc):
  vehicle_id: str = ''
  embedding: NdArray[512]


def add_vehicle(vehicle_id, embedding):
  # Specify your workspace path
  db = HNSWVectorDB[Vehicles](workspace='./vectordb')

  # Index a list of documents with random embeddings
  feature_list = [Vehicles(vehicle_id=vehicle_id, embedding=embedding)]
  db.index(inputs=DocList[Vehicles](feature_list))



def query(embedding):
  db = HNSWVectorDB[Vehicles](workspace='./vectordb')

  # Perform a search query
  query = Vehicles(text='query', embedding=embedding)
  results = db.search(inputs=DocList[Vehicles]([query]), limit=5)
  print(len(results[0].matches))
  # Print out the matches
  for m in results[0].matches:
    print(m)