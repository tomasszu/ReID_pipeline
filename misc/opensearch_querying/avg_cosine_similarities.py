import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

"""

This code computes average cosine similarity variance statistics for each vehicle in each camera from gt+embedding data in the database.
Latest uncommented code though does this for intra-camera (one camera to one camera images) similarities only.


"""



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

def get_embeddings_for_vehicle_and_cam(db, index_name, vehicle_id, cam_id):
    
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

def get_rows_for_vehicle(db, index_name, vehicle_id):
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"vehicle_id": vehicle_id}}
                ]
            }
        }
    }

    response = db.client.search(index=index_name, body=query, size=10000)
    embeds = [hit["_source"] for hit in response["hits"]["hits"]]

    return embeds

def compute_intra_id_stats(db, index_name, vehicle_id):

    # Get all DB rows for this vehicle id, from all cameras, frames etc. 
    all_rows = get_rows_for_vehicle(db, index_name, vehicle_id)

    if not all_rows:
        return None
    
    # Group rows bt camera

    cam_groups = {}
    for row in all_rows:
        cam = row["camera_id"]
        emb = np.array(row["feature_vector"], dtype=np.float32)
        cam_groups.setdefault(cam, []).append(emb)

    # Convert all grouped camera lists to arrays
    for cam in cam_groups:
        cam_groups[cam] = np.stack(cam_groups[cam], axis=0)   # (N, D)

    # Same-camera stats calculation
    same_cam_stats = {}

    for cam, e in cam_groups.items():
        n = e.shape[0]

        if n < 2:
            same_cam_stats[cam] = {"count": n, "mean": None, "std": None, "p0013": None,"p02": None,"p05": None}
            continue

        sim_matrix = cosine_similarity(e, e) # NxN matrix
        # panem tikai upper triangle (jo same vectori tiek reizinaati)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        #flatten prieks percentiileem
        flattened = upper.flatten()

        p0013 = np.percentile(flattened, 0.13)
        p02 = np.percentile(flattened, 2)
        p05 = np.percentile(flattened, 5)

        same_cam_stats[cam] = {"count": n, "mean": float(upper.mean()), "std": float(upper.std()), "p0013": float(p0013),"p02": float(p02),"p05": float(p05)}

    return same_cam_stats



def main():

    db = Opensearch_db("localhost", 9200, ("admin", "admin"))

    f = open("/home/tomass/tomass/ReID_pipele/misc/opensearch_querying/test.csv", "w", encoding='UTF8', newline='')
    writer = csv.writer(f)
    header = ['vehicle_id', 'camera_id', 'count', 'mean','std', 'p0013','p02', 'p05']
    writer.writerow(header)

    index_name="vehicle_vectors"
    # camera_id1, camera_id2 = "S01c004", "S01c001"
    # camera_ids = f"{camera_id1}_{camera_id2}"

    # Implementation for same camera (intra camera) similarities
    ids = find_all_values_of_a_field(db, index_name)

    for veh_id in ids:

        stats = compute_intra_id_stats(db, index_name, veh_id)

        for cam_id, cam_row in stats.items():

            row = [f'{veh_id}']
            row.append(cam_id)
            row = row + list(cam_row.values())
            writer.writerow(row)



    # Old implementation for inter-camera (Two cameras) similarities - kept for reference
    # for veh_id in ids:
        
    #     embeddings1 = get_embeddings_for_vehicle_id(db, index_name, veh_id, camera_id1)
    #     embeddings2 = get_embeddings_for_vehicle_id(db, index_name, veh_id, camera_id2)
    #     if embeddings1 and embeddings2:

    #         e1 = np.array(embeddings1) # shape: (N1, D)
    #         e2 = np.array(embeddings2) # shape: (N2, D)
    #         sim_matrix = cosine_similarity(e1, e2) # N1xN2 matrix

    #         #upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)] # šo vajag tikai ja ir viens un tas pats array ar sevi.

    #         mean = sim_matrix.mean()
    #         std  = sim_matrix.std()
    #         min = sim_matrix.min()
    #         max = sim_matrix.max()

    #         row = [veh_id, camera_ids, mean, std, min, max]

    #         writer.writerow(row)

    f.close()

if __name__ == "__main__":
    main()