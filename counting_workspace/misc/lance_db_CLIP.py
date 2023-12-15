import numpy as np

import misc.lance_db_init_CLIP as create_db


def add_vehicle(vehicle_id, embedding, intersection, db):
  
  tbl = db.open_table(intersection)

  data = [
      {"vehicle_id": vehicle_id, "vector": embedding, "times_summed": 0},
  ]

  tbl.add(data=data)

def update_vehicle(vehicle_id, embedding, intersection, db):
  tbl = db.open_table(intersection)

  df = (tbl.search(np.zeros(512, dtype= np.float32), vector_column_name="vector")
      .where(f"vehicle_id = '{vehicle_id}'", prefilter=True)
      .select(["vector", "vehicle_id", "times_summed"])
      .limit(1)
      .to_list())
  
  if(df):
    if(df[0]['vehicle_id'] == f'{vehicle_id}'):
      times_summed = df[0]['times_summed'] + 1
      embedding_sum = (embedding + (np.array(df[0]['vector']) / times_summed)) / (1 + (1 / times_summed))
      tbl.update(where=f"vehicle_id = '{vehicle_id}'", values={"vector": embedding_sum, "times_summed": times_summed})
      print(f"{vehicle_id}. embedding updated")
      print(f"1 + {1/times_summed} / {1 + (1 / times_summed)}")
    else:
      add_vehicle(vehicle_id, embedding, intersection, db)
      print(f"{vehicle_id}. embedding added")
  else:
    add_vehicle(vehicle_id, embedding, intersection, db)
    print(f"{vehicle_id}. embedding added")
  
  # print(df)



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
  
def query_for_IDs(embedding, intersection):
  db = create_db._init_(intersection)

  # Perform a search query
  tbl = db.open_table(intersection)

  try:
    df = tbl.search(embedding[0]) \
        .limit(3) \
        .metric("l2") \
        .to_list()
    return df
  except:
    return -1

