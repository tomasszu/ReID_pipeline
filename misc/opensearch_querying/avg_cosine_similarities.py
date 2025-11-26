import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv


sys.path.append('/home/tomass/tomass/ReID_pipele/misc')

from opensearch_logic import Opensearch_db

def find_all_values_of_a_field(db, index_name):

    query = {
        "size": 0,
        "aggs": {
            "unique_vehicle_ids": {
                "terms": {
                    "field": "vehicle_id",
                    "size": 10_000     # increase if you have many unique IDs
                }
            }
        }
    }


    response = db.client.search(index=index_name, body=query)
    ids = [bucket["key"] for bucket in response['aggregations']['unique_vehicle_ids']['buckets']]
    return ids

def get_embeddings_for_vehicle_id(db, index_name, vehicle_id, cam_id):
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"vehicle_id": vehicle_id}},
                    {"term": {"camera_id": cam_id}}
                ]
            }
        }
    }

    response = db.client.search(index=index_name, body=query)
    embeds = [hit["_source"]['feature_vector'] for hit in response["hits"]["hits"]]

    return embeds

def main():

    db = Opensearch_db("localhost", 9200, ("admin", "admin"))

    f = open("/home/tomass/tomass/ReID_pipele/misc/opensearch_querying/intra_id_variance_same_cam.csv", "w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    header = ['vehicle_id', 'camera_id', 'mean_same_cam', 'std_same_cam','min_same_cam', 'max_same_cam']
    writer.writerow(header)

    index_name="vehicle_vectors"
    camera_id = "S01c004"


    ids = find_all_values_of_a_field(db, index_name)
    #[print(id) for id in ids]
    vehicles = {}
    for veh_id in ids:
        
        embeddings = get_embeddings_for_vehicle_id(db, index_name, veh_id,camera_id)

        e = np.array(embeddings) # shape: (N, D)
        sim_matrix = cosine_similarity(e, e) # NxN matrix
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        mean_same_cam = upper.mean()
        std_same_cam  = upper.std()
        min_same_cam = upper.min()
        max_same_cam = upper.max()

        row = [veh_id, camera_id, mean_same_cam, std_same_cam, min_same_cam, max_same_cam]

        writer.writerow(row)

    f.close()
    #


if __name__ == "__main__":
    main()