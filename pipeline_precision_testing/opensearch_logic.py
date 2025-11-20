from opensearchpy import OpenSearch
import time

class Opensearch_db:
    def __init__(self, host, port, auth, index_name="vehicle_vectors"):
        self.host = host
        self.port = port
        self.auth = auth
        self.index_name = index_name

        try:
            # Connect to the OpenSearch instance
            self.client = OpenSearch(
                hosts=[{'host': self.host, 'port': self.port}],
                http_auth=self.auth,  # If authentication is enabled
                use_ssl=False,  # Set to True if you're using HTTPS
                verify_certs=False,  # Set to True if you have valid SSL certificates
            )
            
            # Test the connection
            response = self.client.info()
            print("Connected to OpenSearch:", response)
        
        except ConnectionError as e:
            # Handle connection errors, such as network issues or OpenSearch being down
            print(f"Error: Unable to connect to OpenSearch at {self.host}:{self.port}.")
            print(f"Exception: {e}")
            quit(1)
        
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred: {e}\n")

    def insert(self, vehicle_id, feature_vector, cam_id, times_summed=0):
        """
        Inserts a vector document into the OpenSearch index.
        """
        if self.client is None:
            print("Not connected to OpenSearch.")
            return
        
        # --- Sanitize vector ---
        if feature_vector is None:
            print(f"[WARN] Skipping insert for vehicle {vehicle_id}: feature_vector is None\n")
            return        

        
        document = {
            "vehicle_id": vehicle_id,  # Store the actual vehicle ID as a field
            "feature_vector": feature_vector,
            "cam_id": cam_id,
            "timestamp": int(time.time())  # <-- add UNIX epoch timestamp
        }

        try:
            response = self.client.index(index=self.index_name, body=document, refresh=True)
            print(f"Inserted document, vehicle id {vehicle_id}: {response}\n")
        except Exception as e:
            print(f"Error inserting document {vehicle_id}: {e}\n")

        
    def delete_old(self, max_age_seconds=60):
        """
        Deletes documents older than `max_age_seconds`.
        """
        if self.client is None:
            return
        
        cutoff = int(time.time()) - max_age_seconds
        query = {
            "query": {
                "range": {
                    "timestamp": {"lt": cutoff}
                }
            }
        }

        try:
            resp = self.client.delete_by_query(index=self.index_name, body=query, refresh=True)
            deleted = resp.get("deleted", 0)
            if deleted > 0:
                print(f"Deleted {deleted} old documents.\n")
        except Exception as e:
            print(f"Error deleting old docs: {e}\n")

    def query_vector(self, query_vector, cam_id, k=1, threshold=0.6): ##### !!!!!!! Pēdējais šeit, liekas, ka nomainīju lai atgriež tikai 1 result !!!!!!
        """
        Performs a k-NN search on the stored vectors using cosine similarity.
        """
        same_camera_penalty = 0.0  # penalty for same camera queries
        if self.client is None:
            print("Not connected to OpenSearch.")
            return
        
        query = {
            "size": k,
            "query": {
                "knn": {
                    "feature_vector": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=query)
            hits = response.get("hits", {}).get("hits", [])

            # Filter by threshold
            filtered = [(hit["_source"]["vehicle_id"], hit["_score"]) 
                        for hit in hits if ((hit["_source"]["cam_id"] == cam_id and hit["_score"] > threshold + same_camera_penalty) or (hit["_source"]["cam_id"] != cam_id and hit["_score"] > threshold))]

            return filtered if filtered else None
        except Exception as e:
            print(f"Error querying vector: {e}\n")
            return []
        
    def query_id_exists(self, vehicle_id):
        """
        Checks if a vehicle_id exists in the index.
        """
        if self.client is None:
            print("Not connected to OpenSearch.")
            return False
        
        query = {
            "query": {
                "term": {
                    "vehicle_id": vehicle_id
                }
            }
        }

        try:
            response = self.client.search(index=self.index_name, body=query)
            hits = response.get("hits", {}).get("total", {}).get("value", 0)
            return hits > 0
        except Exception as e:
            print(f"Error querying vehicle_id existence: {e}\n")
            return False

