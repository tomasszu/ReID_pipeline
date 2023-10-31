from docarray import BaseDoc
from docarray.typing import NdArray
import numpy as np
from vectordb import HNSWVectorDB
import os
import sys

class Vehicles(BaseDoc):
  vehicle_id: str = ''
  embedding: NdArray[512]

def _init_():
  if(os.path.exists(os.path.join(sys.path[0], '../vectordb/')) is False):
    HNSWVectorDB[Vehicles](workspace=os.path.join(sys.path[0], '../vectordb/'))
  #db.index()
