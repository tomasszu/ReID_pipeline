import sys
sys.path.append('/home/tomass/tomass/ReID_pipele/misc')

from opensearch_logic import Opensearch_db
from IntraVehicleSimAnalyzer import IntraVehicleAnalyzer
from InterVehicleSimAnalyzer import InterVehicleAnalyzer

db = Opensearch_db("localhost", 9200, ("admin", "admin"))

# FOR INTRA-VEHICLE SIMILARITY ANALYSIS (similarities inside vehicle id clusters)
#---------------------------------------------------------------------------------

# an = IntraVehicleAnalyzer(db, "vehicle_vectors")
# an.load_all_embeddings()
# an.compute_stats()

# an.save_camera_stats_csv("misc/opensearch_querying/results/same_camera_similarity_variance_filtered.csv")
# an.save_per_vehicle_stats_csv("misc/opensearch_querying/results/per_vehicle_same_camera_similarity_variance_filtered.csv")
# an.save_cross_camera_stats_csv("misc/opensearch_querying/results/cross_camera_similarity_variance_filtered.csv")

# FOR INTER-VEHICLE SIMILARITY ANALYSIS (similarities across vehicle id clusters)
#---------------------------------------------------------------------------------

an = InterVehicleAnalyzer(db, "vehicle_vectors")
an.load_all_embeddings()
an.compute_stats()

# an.save_per_vehicle_stats_csv("misc/opensearch_querying/results/per_vehicle_inter_similarity_variance.csv")
# an.save_pos_stats_csv("misc/opensearch_querying/results/positive_inter_similarity_variance.csv")
# an.save_neg_stats_csv("misc/opensearch_querying/results/negative_inter_similarity_variance.csv")

an.plot_similarity_distributions2()