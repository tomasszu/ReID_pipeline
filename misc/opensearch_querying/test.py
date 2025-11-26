import sys


sys.path.append('/home/tomass/tomass/ReID_pipele/misc')

from opensearch_logic import Opensearch_db




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
                    "field": "vehicle_id",
                    "size": 10_000     # increase if you have many unique IDs
                }
            }
        }
    }


    response = db.client.search(index=index_name, body=query)
    ids = [bucket["key"] for bucket in response['aggregations']['unique_vehicle_ids']['buckets']]

    print(ids)

def main():

    db = Opensearch_db("localhost", 9200, ("admin", "admin"))

    index_name="vehicle_vectors"

    #query_two_fields(db, index_name)

    find_all_values_of_a_field(db, index_name)


if __name__ == "__main__":
    main()