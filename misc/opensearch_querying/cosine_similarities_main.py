import sys
sys.path.append('/home/tomass/tomass/ReID_pipele/misc')

from opensearch_logic import Opensearch_db
from IntraVehicleSimAnalyzer import IntraVehicleAnalyzer

db = Opensearch_db("localhost", 9200, ("admin", "admin"))


an = IntraVehicleAnalyzer(db, "vehicle_vectors")
an.load_all_embeddings()
an.compute_stats()

an.save_camera_stats_csv("misc/opensearch_querying/results/same_camera_similarity_variance.csv")
an.save_per_vehicle_stats_csv("misc/opensearch_querying/results/per_vehicle_same_camera_similarity_variance.csv")
an.save_cross_camera_stats_csv("misc/opensearch_querying/results/cross_camera_similarity_variance.csv")