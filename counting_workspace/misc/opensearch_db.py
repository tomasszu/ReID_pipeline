from opensearchpy import OpenSearch

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
        
        except Exception as e:
            # Catch any other exceptions
            print(f"An unexpected error occurred: {e}")

    def insert(self, vehicle_id, feature_vector, times_summed=0):
        """
        Inserts a vector document into the OpenSearch index.
        """
        if self.client is None:
            print("Not connected to OpenSearch.")
            return
        
        document = {
            "vehicle_id": vehicle_id,  # Store the actual vehicle ID as a field
            "feature_vector": feature_vector,
            "times_summed": times_summed
        }

        try:
            response = self.client.index(index=self.index_name, body=document, refresh=True)
            #print(f"Inserted document {vehicle_id}: {response}")
        except Exception as e:
            print(f"Error inserting document {vehicle_id}: {e}")


    def query_vector(self, query_vector, k=5):
        """
        Performs a k-NN search on the stored vectors using cosine similarity.
        """
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
            return [(hit["_source"]["vehicle_id"], hit["_score"]) for hit in hits]  # Return vehicle_id instead of doc_id
        except Exception as e:
            print(f"Error querying vector: {e}")
            return []

