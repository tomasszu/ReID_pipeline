

class ResultsTally:
    def __init__(self, database):
        self.db = database

    def display_results(self):
        primary_precision = (self.db.primary_hits_count / self.db.total_primary_queries * 100) if self.db.total_primary_queries > 0 else 0
        secondary_precision = (self.db.secondary_hits_count / self.db.total_secondary_queries * 100) if self.db.total_secondary_queries > 0 else 0

        print("----- Precision Testing Results -----")
        print(f"Primary Queries: {self.db.total_primary_queries}, Primary Hits: {self.db.primary_hits_count}, Primary Precision: {primary_precision:.2f}%")
        print(f"Secondary Queries: {self.db.total_secondary_queries}, Secondary Hits: {self.db.secondary_hits_count}, Secondary Precision: {secondary_precision:.2f}%")
        print("-------------------------------------")