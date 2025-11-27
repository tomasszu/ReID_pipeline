import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import csv
from opensearchpy.helpers import scan

class IntraVehicleAnalyzer:

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

        self.per_vehicle_stats = defaultdict(dict)
        # self.per_vehicle_stats[veh_id][cam_id] = {mean, std, p02, ...}

        self.per_camera_stats = {}
        # self.per_camera_stats[cam_id] = {...}

        self.per_camera_raw = defaultdict(list)
        # self.per_camera_raw[cam_id] = [all raw similarity vals]

        self.cross_camera_raw = defaultdict(list)
        # self.cross_camera_raw[(camA, camB)] = [raw sim values]


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

            veh = src["vehicle_id"]
            cam = src["camera_id"]
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

        for veh_id in vehicle_ids:

            cams = self.embeddings[veh_id].keys()

            # 1) SAME CAMERA STATS -----------------------------------------
            for cam_id in cams:
                arr = np.array(self.embeddings[veh_id][cam_id])
                if len(arr) < 2:
                    continue

                sim = cosine_similarity(arr, arr)
                vals = sim[np.triu_indices_from(sim, k=1)]

                # store per-vehicle stats
                self.per_vehicle_stats[veh_id][cam_id] = self._compute_stat_dict(vals)

                # store raw vals per camera
                self.per_camera_raw[cam_id].extend(vals.tolist())

            # 2) CROSS CAMERA STATS ----------------------------------------
            cam_list = list(cams)
            for i in range(len(cam_list)):
                for j in range(i + 1, len(cam_list)):
                    c1, c2 = cam_list[i], cam_list[j]

                    arr1 = np.array(self.embeddings[veh_id][c1])
                    arr2 = np.array(self.embeddings[veh_id][c2])

                    sim = cosine_similarity(arr1, arr2).flatten()
                    self.cross_camera_raw[(c1, c2)].extend(sim.tolist())

        # AFTER PROCESSING ALL VEHICLES -----------------------------------
        self._compute_per_camera_stats()
        self._compute_cross_camera_stats()


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
        }


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
    # CSV EXPORT
    # ---------------------------------------------------------------------

    def save_camera_stats_csv(self, path):
        """
        Saves exactly one row per camera.
        """
        fieldnames = ["camera_id", "count", "mean", "std", "p0013", "p02", "p05"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for cam, stats in self.per_camera_stats.items():
                row = {"camera_id": cam, **stats}
                w.writerow(row)

    def save_cross_camera_stats_csv(self, path):
        """
        Saves one row per (camA, camB) pair.
        """
        fieldnames = ["camera_a", "camera_b", "count", "mean", "std", "p0013", "p02", "p05"]
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
        fieldnames = ["vehicle_id", "camera_id", "count", "mean", "std", "p0013", "p02", "p05"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for veh_id, cam_dict in self.per_vehicle_stats.items():
                for cam_id, stats in cam_dict.items():
                    row = {"vehicle_id": veh_id, "camera_id": cam_id, **stats}
                    w.writerow(row)
