import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import csv
from opensearchpy.helpers import scan
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from scipy.stats import entropy

class InterVehicleAnalyzer:

    def __init__(self, db, index_name):
        """
        db: OpenSearch client wrapper
        index_name: index with documents (vehicle_id, cam_id, embedding, ...)
        """
        self.db = db
        self.index_name = index_name

        # data structures
        self.embeddings = defaultdict(lambda: defaultdict(list))
        # self.embeddings[vehicle_id][camera_id] = [emb1, emb2, ...]

        self.per_vehicle_stats = {}

        self.all_positive_raw = []

        self.all_negative_raw = []

        self.same_cam_pos_raw = []
        # self.same_cam_pos_raw = [all raw similarity vals]

        self.cross_cam_pos_raw = []

        self.same_cam_neg_raw = []

        self.cross_cam_neg_raw = []

        self.per_camera_raw = defaultdict(list)
        # self.per_camera_raw[cam_id] = [all raw similarity vals]

        self.cross_camera_raw = defaultdict(list)
        # self.cross_camera_raw[(camA, camB)] = [raw sim values]

    # DATA FILTER FUNCTIONS
    #---------------------------------------------------------------------

    def calculate_center_point(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        return (center_x, center_y)
    
    def filter_cp(self, center_point, cam_id):
        x, y = center_point

        if cam_id == "S01c004":
            # bound bottom left
            bl = (0, 1000)
            # bound top right
            tr = (1750, 320)
            if bl[0] <= x <= tr[0] and tr[1] <= y <= bl[1]:
                return True
        elif cam_id == "S01c001":
            bl = (200, 900)
            tr = (1750, 320)
            if bl[0] <= x <= tr[0] and tr[1] <= y <= bl[1]:
                return True
        elif cam_id == "S01c002":
            bl = (200, 900)
            tr = (1750, 320)
            if bl[0] <= x <= tr[0] and tr[1] <= y <= bl[1]:
                return True
        elif cam_id == "S01c003":
            bl = (200, 900)
            tr = (1750, 270)
            if bl[0] <= x <= tr[0] and tr[1] <= y <= bl[1]:
                return True

        return False


    # ---------------------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------------------

    def load_all_embeddings(self):
        """
        Fetch all docs from OpenSearch, group embeddings by (veh_id, cam_id).
        """

        scroll = scan(
            self.db.client,
            index=self.index_name,
            query={"query": {"match_all": {}}},
            scroll="2m",
            size=500
        )

        for doc in scroll:
            src = doc["_source"]

            bbox = src["bbox"]
            cam = src["camera_id"]
            center_point = self.calculate_center_point(bbox)

            if self.filter_cp(center_point, cam):
                veh = src["vehicle_id"]            
                emb = np.array(src["feature_vector"], dtype=np.float32)

                self.embeddings[veh][cam].append(emb)




    # ---------------------------------------------------------------------
    # MAIN PROCESSING
    # ---------------------------------------------------------------------

    def compute_stats(self):
        """
        Compute:
        - per-vehicle per-camera stats
        - pooled camera stats (exact)
        - cross-camera stats (exact)
        """
        vehicle_ids = list(self.embeddings.keys())
        camera_ids = self._collect_camera_ids()

        for vi, veh_id in enumerate(vehicle_ids):
            print(f"[DEBUG] Processing vehicle id: {veh_id}")

            cams = self.embeddings[veh_id].keys()

            # 1) SAME CAMERA STATS -----------------------------------------
            # for cam_id in cams:
            #     arr = np.array(self.embeddings[veh_id][cam_id])
            #     if len(arr) < 2:
            #         continue

            #     sim = cosine_similarity(arr, arr)
            #     vals = sim[np.triu_indices_from(sim, k=1)]

            #     # store per-vehicle stats
            #     self.per_vehicle_stats[veh_id][cam_id] = self._compute_stat_dict(vals)

            #     # store raw vals per camera
            #     self.per_camera_raw[cam_id].extend(vals.tolist())

            # # 2) CROSS CAMERA STATS ----------------------------------------
            # cam_list = list(cams)
            # for i in range(len(cam_list)):
            #     for j in range(i + 1, len(cam_list)):
            #         c1, c2 = cam_list[i], cam_list[j]

            #         arr1 = np.array(self.embeddings[veh_id][c1])
            #         arr2 = np.array(self.embeddings[veh_id][c2])

            #         sim = cosine_similarity(arr1, arr2).flatten()
            #         self.cross_camera_raw[(c1, c2)].extend(sim.tolist())

            # # 3) SAME VEHICLE STATS (Positives) ACROSS ALL CAMERAS -----------------------------
            # all_embs = []
            # for cam_id in cams:
            #     all_embs.extend(self.embeddings[veh_id][cam_id])
            # all_arr =  np.array(all_embs)
            # if len(all_arr) < 2:
            #     continue
            # sim = cosine_similarity(all_arr, all_arr)
            # vals = sim[np.triu_indices_from(sim, k=1)]

            # # store per-vehicle stats
            # self.per_vehicle_stats[veh_id] = self._compute_stat_dict(vals)

            # # store all raw positive vals
            # self.all_positive_raw.extend(vals.tolist())


            # # 4) OTHER VEHICLES (Negatives) STATS ACROSS ALL CAMERAS -----------------------------

            # for other_veh_id in vehicle_ids:
            #     if other_veh_id == veh_id:
            #         continue
            #     other_embs = []
            #     for cam_id in self.embeddings[other_veh_id].keys():
            #         other_embs.extend(self.embeddings[other_veh_id][cam_id])
            #     other_arr = np.array(other_embs)
            #     if len(other_arr) < 1:
            #         continue
            #     sim = cosine_similarity(all_arr, other_arr).flatten()
            #     # store all raw negative vals
            #     self.all_negative_raw.extend(sim.tolist())

            # 5) Same camera positives and negatives -----------------------------------

            for cam_id in camera_ids:
                if cam_id in cams:
                    arr = np.array(self.embeddings[veh_id][cam_id])
                    if len(arr) >= 2:
                        # Positives
                        sim = cosine_similarity(arr, arr)
                        vals = sim[np.triu_indices_from(sim, k=1)]
                        self.same_cam_pos_raw.extend(vals.tolist())
                else:
                    continue
                # Negatives
                for vj in range(vi + 1,len(vehicle_ids)):
                    other_veh_id = vehicle_ids[vj]

                    if cam_id in self.embeddings[other_veh_id]:
                        other_arr = np.array(self.embeddings[other_veh_id][cam_id])
                        if len(other_arr) >= 1:
                            sim = cosine_similarity(arr, other_arr).flatten()
                            self.same_cam_neg_raw.extend(sim.tolist())
            


            # 6) Cross camera positives and negatives -----------------------------------

            cam_list = list(cams)
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    c1, c2 = cam_list[i], cam_list[j]

                    arr1 = np.array(self.embeddings[veh_id][c1])
                    arr2 = np.array(self.embeddings[veh_id][c2])

                    sim = cosine_similarity(arr1, arr2).flatten()
                    self.cross_cam_pos_raw.extend(sim.tolist())
            
            compared_cached = set()
            # --- NEGATIVES (different vehicles, cross-camera only, no duplicates) ---
            for camA in cams:
                arrA = np.array(self.embeddings[veh_id][camA])

                for vj in range(vi + 1,len(vehicle_ids)):
                    other_veh_id = vehicle_ids[vj]

                    for camB, other_embs in self.embeddings[other_veh_id].items():

                        # skip same camera comparisons → those belong to same_cam_neg_raw
                        if camA == camB:
                            continue

                        # build symmetric pair key
                        pair_key = tuple(sorted([
                            (veh_id, camA),
                            (other_veh_id, camB)
                        ]))

                        # check duplicates (both directions)
                        if pair_key in compared_cached:
                            continue

                        arrB = np.array(other_embs)
                        if len(arrB) == 0:
                            continue

                        sim = cosine_similarity(arrA, arrB).flatten()
                        self.cross_cam_neg_raw.extend(sim.tolist())

                        compared_cached.add(pair_key)
                
            
                

        # AFTER PROCESSING ALL VEHICLES -----------------------------------
        #commented out if you only want diagrams
        # self._compute_all_positive_stats()
        # self._compute_all_negative_stats()


    # ---------------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------------

    def _collect_camera_ids(self):
        cams = set()
        for veh_id in self.embeddings:
            cams.update(self.embeddings[veh_id].keys())
        return list(cams)


    def _compute_stat_dict(self, vals: np.ndarray):
        return {
            "count": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p0013": float(np.percentile(vals, 0.13)),
            "p02": float(np.percentile(vals, 2)),
            "p05": float(np.percentile(vals, 5)),
            "p10": float(np.percentile(vals, 10)),
        }
    
    def _compute_neg_stat_dict(self, vals: np.ndarray):
        return {
            "count": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p9987": float(np.percentile(vals, 99.87)),
            "p98": float(np.percentile(vals, 98)),
            "p95": float(np.percentile(vals, 95)),
            "p90": float(np.percentile(vals, 90)),
        }
    
    def _compute_all_positive_stats(self):
        """
        Pools all raw vals for positives (same vehicle).
        """
        arr = np.array(self.all_positive_raw)
        self.all_pos_stats = self._compute_stat_dict(arr)

    def _compute_all_negative_stats(self):
        """
        Pools all raw vals for negatives (different vehicles).
        """
        arr = np.array(self.all_negative_raw)
        self.all_neg_stats = self._compute_neg_stat_dict(arr)



    def _compute_per_camera_stats(self):
        """
        Pools all raw vals per camera (exact).
        """
        for cam, raw in self.per_camera_raw.items():
            if len(raw) == 0:
                continue
            arr = np.array(raw)
            self.per_camera_stats[cam] = self._compute_stat_dict(arr)


    def _compute_cross_camera_stats(self):
        """
        Pools all raw cross-camera similarities.
        """
        self.cross_camera_stats = {}
        for pair, raw in self.cross_camera_raw.items():
            if len(raw) == 0:
                continue
            arr = np.array(raw)
            self.cross_camera_stats[pair] = self._compute_stat_dict(arr)

    # ---------------------------------------------------------------------
    # GRAPH EXPORT
    # ---------------------------------------------------------------------


    def plot_similarity_distributions(self):
        # uses 3.) and 4.) from compute_stats()
        pos_vals = self.all_positive_raw
        neg_vals = self.all_negative_raw
        pos_vals = np.array(pos_vals)
        neg_vals = np.array(neg_vals)

        # Optional smooth curves
        pos_kde = gaussian_kde(pos_vals)
        neg_kde = gaussian_kde(neg_vals)

        x = np.linspace(-0.25, 1, 400)

        plt.figure(figsize=(10, 5))

        # Histograms
        plt.hist(pos_vals, bins=50, density=True, alpha=0.4, label="Positive (same vehicle)")
        plt.hist(neg_vals, bins=50, density=True, alpha=0.4, label="Negative (different vehicle)")

        # Smooth lines
        plt.plot(x, pos_kde(x), linewidth=2)
        plt.plot(x, neg_kde(x), linewidth=2)

        plt.xlabel("Cosine similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("misc/opensearch_querying/results/inter_vehicle_similarity_distribution.png")
        plt.show()

    def plot_similarity_distributions2(self):
        # This one includes the same-camera and cross-camera distributions separately
        # aka using the distributions for same-cam positives, cross-cam positives, same-cam negatives, cross-cam negatives
        # Uses 5.) and 6.) from compute_stats()

        pos_vals_same = np.array(self.same_cam_pos_raw)
        pos_vals_cross = np.array(self.cross_cam_pos_raw)
        neg_vals_same = np.array(self.same_cam_neg_raw)
        neg_vals_cross = np.array(self.cross_cam_neg_raw)

        # Optional smooth curves
        # print("[DEBUG] Computing KDEs...")
        # pos_kde1 = gaussian_kde(pos_vals_same)
        # pos_kde2 = gaussian_kde(pos_vals_cross)
        # neg_kde1 = gaussian_kde(neg_vals_same)
        # neg_kde2 = gaussian_kde(neg_vals_cross)

        x = np.linspace(-0.25, 1, 400)

        plt.figure(figsize=(10, 5))

        # Histograms
        print("[DEBUG] Plotting histograms...")
        plt.hist(pos_vals_same, bins=50, density=True, alpha=0.4, label="Positive (same camera)")
        plt.hist(pos_vals_cross, bins=75, density=True, alpha=0.4, label="Positive (different camera)")
        plt.hist(neg_vals_same, bins=100, density=True, alpha=0.4, label="Negative (same camera)")
        plt.hist(neg_vals_cross, bins=100, density=True, alpha=0.4, label="Negative (different camera)")

        # Smooth lines
        print("[DEBUG] Plotting KDE lines...")
        # plt.plot(x, pos_kde1(x), linewidth=2)
        # plt.plot(x, pos_kde2(x), linewidth=2)
        # plt.plot(x, neg_kde1(x), linewidth=2)
        # plt.plot(x, neg_kde2(x), linewidth=2)

        print("[DEBUG] Finalizing plot...")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("misc/opensearch_querying/results/inter_vehicle_similarity_distribution2.png")
        plt.show()

    
    # ---------------------------------------------------------------------
    # PRINT STATS
    # ---------------------------------------------------------------------

    def print_KL_divergence(self,a,b, bins=100):

        # Printing Kullbach-Leibner divergence between two probability distributions

        p_hist, bin_edges = np.histogram(a, bins=bins, range=(-0.25, 1.0), density=True)
        q_hist, _ = np.histogram(b, bins=bin_edges, density=True)

        # avoid zero values
        p_hist += 1e-9
        q_hist += 1e-9

        kl_value = entropy(p_hist, q_hist) # KL divergence D_KL(P || Q)

        print(f"KL Divergence: {kl_value}")

    def sample_info(self, name, arr):
        # print sample info abt a distribution array values
        arr = np.asarray(arr)
        print(name)
        print(" count:", arr.size)
        print(" unique values:", np.unique(arr).size)
        vals, counts = np.unique(arr.round(6), return_counts=True)  # round -> group near-equal
        top = sorted(zip(counts, vals), reverse=True)[:10]
        print(" top repeat values (count, value):", top)
        print(" min/max:", arr.min(), arr.max())
        print(" mean/std:", arr.mean(), arr.std())
        print(" percentiles 1,5,10,25,50,75,90,95,99:", np.percentile(arr, [1,5,10,25,50,75,90,95,99]))
        print()



    # ---------------------------------------------------------------------
    # CSV EXPORT
    # ---------------------------------------------------------------------

    def save_pos_stats_csv(self, path):
        """
        Saves exactly one row per camera.
        """
        fieldnames = ["count", "mean", "std", "p0013", "p02", "p05", "p10"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            row = self.all_pos_stats
            w.writerow(row)

    def save_neg_stats_csv(self, path):
        """
        Saves exactly one row per camera.
        """
        fieldnames = ["count", "mean", "std", "p9987", "p98", "p95", "p90"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            row = self.all_neg_stats
            w.writerow(row)

    def save_cross_camera_stats_csv(self, path):
        """
        Saves one row per (camA, camB) pair.
        """
        fieldnames = ["camera_a", "camera_b", "count", "mean", "std", "p0013", "p02", "p05", "p10"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for (c1, c2), stats in self.cross_camera_stats.items():
                row = {"camera_a": c1, "camera_b": c2, **stats}
                w.writerow(row)

    def save_per_vehicle_stats_csv(self, path):
        """
        Saves one row per (veh_id, cam_id) pair.
        """
        fieldnames = ["vehicle_id", "count", "mean", "std", "p0013", "p02", "p05", "p10"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for veh_id, stats in self.per_vehicle_stats.items():
                row = {"vehicle_id": veh_id, **stats}
                w.writerow(row)
