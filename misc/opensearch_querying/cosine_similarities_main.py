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

an = InterVehicleAnalyzer(db, "vehicle_vectors2")
an.load_all_embeddings()
an.compute_stats()

# an.save_per_vehicle_stats_csv("misc/opensearch_querying/results/per_vehicle_inter_similarity_variance.csv")
# an.save_pos_stats_csv("misc/opensearch_querying/results/positive_inter_similarity_variance.csv")
# an.save_neg_stats_csv("misc/opensearch_querying/results/negative_inter_similarity_variance.csv")

an.plot_similarity_distributions2()

# PRINT STATS
# print("Divergence starp same un cross cam negative similarities")
#an.print_KL_divergence(an.same_cam_neg_raw, an.cross_cam_neg_raw)
print("Seperation metric between same and cross cam negative similarities")
an.print_separation_metric(an.cross_cam_neg_raw, an.same_cam_neg_raw)
print("ROC AUC Score between same and cross cam negative similarities")
an.print_roc_auc_score(an.cross_cam_neg_raw, an.same_cam_neg_raw)
print("Percentile gaps between same and cross cam negative similarities")
an.print_percentile_gap(an.cross_cam_neg_raw, an.same_cam_neg_raw)
an.print_percentile_gap(an.cross_cam_neg_raw, an.same_cam_neg_raw, p=10)


# print("Divergence starp positive un negative similarities distributions (cross camera)")
#an.print_KL_divergence(an.cross_cam_neg_raw, an.cross_cam_pos_raw)
print("Seperation metric between positive and negative similarities distributions (cross camera)")
an.print_separation_metric(an.cross_cam_neg_raw, an.cross_cam_pos_raw)
print("ROC AUC Score between positive and negative similarities distributions (cross camera)")
an.print_roc_auc_score(an.cross_cam_neg_raw, an.cross_cam_pos_raw)
print("Percentile gaps between positive and negative similarities distributions (cross camera)")
an.print_percentile_gap(an.cross_cam_neg_raw, an.cross_cam_pos_raw)
an.print_percentile_gap(an.cross_cam_neg_raw, an.cross_cam_pos_raw, p=10)

print("Sample stats:")
an.sample_info("pos_cross", an.cross_cam_pos_raw)
an.sample_info("pos_same", an.same_cam_pos_raw)
an.sample_info("neg_cross", an.cross_cam_neg_raw)
an.sample_info("neg_same", an.same_cam_neg_raw)