from opensearch_logic import Opensearch_db


class Database:

    def __init__(self):

        self.db = Opensearch_db("localhost", 9200, ("admin", "admin"))

        self.list_of_existing_ids = set()

    def insert(self, id, vector, cam_id):

        self.db.insert(id, vector, cam_id)