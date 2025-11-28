import csv

rows_cross = []
rows_same = {}
with open('misc/opensearch_querying/results/cross_camera_similarity_variance_filtered.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        print(', '.join(row))
        row = row[0].split(',')
        rows_cross.append(row)
    
    rows_cross = rows_cross[1:]  # skip header

with open('misc/opensearch_querying/results/same_camera_similarity_variance_filtered.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        print(', '.join(row))
        row = row[0].split(',')
        rows_same[row[0]] = row[1:]
    
    rows_same.pop('camera_id')  # skip header

print("Calculating changes...")
print(rows_cross)
print("-----")
print(rows_same)

with open('misc/opensearch_querying/results/same_to_cross_cam_change.csv', 'w', newline='') as csvfile:
    fieldnames = ["camera_a", "camera_b", "mean_change", "std_change","p0013_change","p02_change","p05_change","p10_change"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in rows_cross:
        cam_a = row[0]
        cam_b = row[1]

        same_stats1 = rows_same.get(cam_a, None)
        same_stats2 = rows_same.get(cam_b, None)

        if same_stats1 is None or same_stats2 is None:

            print(f"Skipping pair ({cam_a}, {cam_b}) due to missing same-camera stats.")
            continue
        mean_change = float(row[3]) - (float(same_stats1[1]) + float(same_stats2[1])) / 2
        std_change = float(row[4]) - (float(same_stats1[2]) + float(same_stats2[2])) / 2
        p0013_change = float(row[5]) - (float(same_stats1[3]) + float(same_stats2[3])) / 2
        p02_change = float(row[6]) - (float(same_stats1[4]) + float(same_stats2[4])) / 2
        p05_change = float(row[7]) - (float(same_stats1[5]) + float(same_stats2[5])) / 2
        p10_change = float(row[8]) - (float(same_stats1[6]) + float(same_stats2[6])) / 2
        writer.writerow({
            "camera_a": cam_a,
            "camera_b": cam_b,
            "mean_change": mean_change,
            "std_change": std_change,
            "p0013_change": p0013_change,
            "p02_change": p02_change,
            "p05_change": p05_change,
            "p10_change": p10_change
        })
print("Done.")

    

