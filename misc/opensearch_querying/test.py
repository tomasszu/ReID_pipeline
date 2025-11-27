import sys


sys.path.append('/home/tomass/tomass/ReID_pipele/misc')

from opensearch_logic import Opensearch_db
from opensearchpy.helpers import scan

from collections import defaultdict
import numpy as np





def query_two_fields(db, index_name):

    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"vehicle_id": 1}},
                    {"term": {"camera_id": "S01c004"}}
                ]
            }
        }
    }


    response = db.client.search(index=index_name, body=query)
    hits = response.get("hits", {}).get("hits", {})

    filtered = [hit["_source"] for hit in hits]
    print(filtered)

def find_all_values_of_a_field(db, index_name):

    query = {
        "size": 0,
        "aggs": {
            "unique_vehicle_ids": {
                "terms": {
                    "field": "camera_id",
                    "size": 10_000     # increase if you have many unique IDs
                }
            }
        }
    }


    response = db.client.search(index=index_name, body=query)
    ids = [bucket["key"] for bucket in response['aggregations']['unique_vehicle_ids']['buckets']]

    print(ids)

def gather_and_filter_rows(db, index_name):

    embeddings = {}

    scroll = scan(
            db.client,
            index=index_name,
            query={"query": {"match_all": {}}},
            scroll="2m",
            size=500
        )
    iter = 0
    for doc in scroll:
        src = doc["_source"]
        
        bbox = src["bbox"]
        veh = src["vehicle_id"]
        cam = src["camera_id"]
        frame = np.array(src["frame_id"], dtype=np.float32)

        embeddings[iter] = ((frame, cam, veh, bbox))
        iter += 1

    print(f"Loaded embeddings for {len(embeddings)} vehicles.")

def main():

    db = Opensearch_db("localhost", 9200, ("admin", "admin"))

    index_name="vehicle_vectors"

    #query_two_fields(db, index_name)

    #find_all_values_of_a_field(db, index_name)

    gather_and_filter_rows(db, index_name)


if __name__ == "__main__":
    main()