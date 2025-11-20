from opensearch_logic import Opensearch_db


class Database:

    def __init__(self):

        self.db = Opensearch_db("localhost", 9200, ("admin", "admin"))

        self.list_of_existing_ids = set()

        # Test metric counts:
        self.primary_hits_count = 0
        self.secondary_hits_count = 0
        self.total_primary_queries = 0
        self.total_secondary_queries = 0

    def insert(self, id, vector, cam_id):

        self.db.insert(id, vector, cam_id)

    def query(self, id, vector, cam_id):


        if id in self.list_of_existing_ids:
            print(f"{id} found in local cache\n")
            self.insert(id, vector, cam_id)
        else:
            print(f"{id} not found in local cache, querying database\n")
            
            filtered_result = self.db.query_vector(vector, cam_id)
            id_exists = self.db.query_id_exists(id)

            if filtered_result:
                print(f"{id} from {cam_id}. cam found in database as: {filtered_result[0]}\n")
                if int(filtered_result[0]) == int(id):
                    self.insert(id, vector, cam_id)
                    self.secondary_hits_count += 1
                    self.total_secondary_queries += 1
                else:
                    if id_exists:
                        self.total_secondary_queries += 1
                    else:
                        self.total_primary_queries += 1
            else:
                print(f"{id} from {cam_id}. cam not found in database\n")
                if id_exists:
                    self.total_secondary_queries += 1
                else:
                    self.total_primary_queries += 1
                    print(f"Inserting as new id: {id}\n")
                    self.insert(id, vector, cam_id)
                    self.primary_hits_count += 1
                    self.list_of_existing_ids.add(id)

                

        #return (composite_id, cam_id) if corrected_id is None else (corrected_id, cam_id)