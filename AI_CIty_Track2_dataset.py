import numpy as np 
import pandas as pd
import os
import random
import torch
from torchvision import transforms
from PIL import Image
from vehicle_reid_repo.vehicle_reid.load_model import load_model_from_opts
import matplotlib.pyplot as plt
import csv
from lxml import etree

DATASET = "/home/tomass/tomass/data/(Cityflow)AIC21_Track2_ReID_full/AIC21_Track2_ReID/"
DATA_ROOT = "../data/"

#ids to remove from training set, that are used in the testing videos later on.
testing_ids = [1, 34, 7, 4, 35, 54, 8, 37, 9, 36, 11, 12, 13, 14, 15, 38, 16, 17, 55, 39, 18, 19, 20, 41, 21, 22, 24, 40, 26, 27, 6, 29, 23, 31, 32, 28, 33, 44, 45, 46, 47, 49, 48, 50, 51, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 68, 67, 69, 73, 74, 52, 53, 75, 76, 77, 79, 80, 82, 83, 85, 89, 71, 90, 72, 91, 92, 93, 94]


def convert_to_xml(input_file, output_file):
    input_file = DATASET + input_file
    output_file = DATASET + output_file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="gb2312" ?>\n')
        f.write('<TrainingImages Version="1.0">\n')
        f.write('    <Items number="' + str(len(lines)) + '">\n')

        for idx, line in enumerate(lines, start=1):
            image_names = line.strip().split()
            for image_name in image_names:
                f.write(f'        <Item imageName="{image_name}" vehicleID="{idx:04d}" cameraID="c000" />\n')

        f.write('    </Items>\n')
        f.write('</TrainingImages>\n')

# Usage example
# convert_to_xml('test_track.txt', 'test_label.xml')
        

def convert_xml_to_csv(xml_file, csv_file, dataset_path, folder_name):
    parser = etree.XMLParser(encoding='gb2312')
    tree = etree.parse(xml_file, parser=parser)
    root = tree.getroot()

    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['path', 'id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in root.findall('.//Item'):
            image_name = item.get('imageName')
            vehicle_id = item.get('vehicleID')
            path = f"{dataset_path}{folder_name}/{image_name}"
            writer.writerow({'path': path, 'id': vehicle_id})

# Usage example
# xml_file = DATASET + 'train_label.xml'
# csv_file = DATASET + 'train_label.csv'
# dataset_path = DATASET  # Adjust this according to your dataset path
# folder_name = 'image_train'
# convert_xml_to_csv(xml_file, csv_file, dataset_path, folder_name)

def split_csv_dataset(csv_in, dataset_path, folder_name):
    df = pd.read_csv(csv_in)
    df.columns = ["path", "id"]
    df = df[["path", "id"]]
    df["path"] = df["path"].apply(lambda x: os.path.join(f"{dataset_path}{folder_name}/", x))



    # fix the seed to always generate the same sets
    random.seed(42)

    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))

    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    len(train_df), len(val_df)


    # print(train_df)
    # print(val_df)

    train_df.to_csv(f"{dataset_path}/" + "train_label_split.csv", index=False)
    val_df.to_csv(f"{dataset_path}/" + "val_label_split.csv", index=False)

split_csv_dataset(DATASET + 'train_label_filtered.csv', DATASET, 'image_train')

def remove_idx_from_csv_dataset(csv_in, dataset_path, removable_ids):
    
    # Read the CSV file
    df = pd.read_csv(csv_in)

    # Remove padding zeros from the 'id' column
    df['id'] = df['id'].astype(str).str.lstrip('0')

    # Convert 'id' column to integer
    df['id'] = df['id'].astype(int)

    # Filter out rows with IDs present in testing_ids list
    df = df[~df['id'].isin(removable_ids)]

    # Save the modified DataFrame back to CSV
    df.to_csv(f"{dataset_path}/" +"train_label_filtered.csv", index=False)

# remove_idx_from_csv_dataset(DATASET + 'train_label.csv', DATASET, testing_ids)
    



#TXT anotaciju failu parversana uz CSV (tikai imdir un vehicle ID laukus atstaj)

# def txt_labels_to_csv(txt_path, out_path, img_dir):
#     df = pd.read_csv(txt_path, sep=" ")
#     df.columns = ["path", "id", "cam"]
#     df = df[["path", "id"]]
#     df["path"] = df["path"].apply(lambda x: os.path.join(img_dir, x))
#     df.to_csv(out_path, index=False)
    
# txt_labels_to_csv(VRIC + "vric_gallery.txt", VRIC + "vric_gallery.csv", "VRIC/gallery_images/")
# txt_labels_to_csv(VRIC + "vric_probe.txt", VRIC + "vric_query.csv", "VRIC/probe_images/")

#TXT anotaciju failu parversana uz CSV un sadalisana test un validation datasetos (tikai imdir un vehicle ID laukus atstaj)

# def train_data_split_and_to_csv(train_ann, train_out_path, val_out_path, img_dir):

#     df = pd.read_csv(train_ann, sep=" ")
#     df.columns = ["path", "id", "cam"]
#     df = df[["path", "id"]]
#     df["path"] = df["path"].apply(lambda x: os.path.join(img_dir, x))

#     random.seed(42)

#     train_size = int(0.75 * len(df))
#     val_size = len(df) - train_size
#     val_idxes = random.sample(range(len(df)), val_size)
#     train_idxes = list(set(range(len(df))) - set(val_idxes))

#     train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]

#     train_df.to_csv(train_out_path, index=False)
#     val_df.to_csv(val_out_path, index=False)

# train_data_split_and_to_csv(VRIC + "vric_train.txt", VRIC + "vric_train.csv", VRIC + "vric_val.csv", "VRIC/train_images/")




# device = "cuda"

# model = load_model_from_opts("vehicle_reid_repo/vehicle_reid/model/result6/opts.yaml", ckpt="vehicle_reid_repo/vehicle_reid/model/result6/net_19.pth", remove_classifier=True)
# model.eval()
# model.to(device)
# # X = torch.randn(32, 3, 224, 224).to(device)
# # X = model(X)
# # print(X.shape)

# random.seed(2)


# #Image transforms probably adapted from vehicle Re-ID model code
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# #Selecting a random query image

# query_df = pd.read_csv(DATA_ROOT + "VRIC/vric_query.csv")
# query_path, query_label = query_df.loc[random.choice(range(len(query_df)))]
# query_image = Image.open(DATA_ROOT + query_path)
# X_query = torch.unsqueeze(data_transforms(query_image), 0).to(device)
# #print(X_query.shape)

# N_GALLERY = 32

# #Ielasam query datafailu
# gallery_df = pd.read_csv(DATA_ROOT + "VRIC/vric_gallery.csv")

# # make sure we choose at least one with the same id as the query
# same_id_idxes = list(gallery_df[gallery_df["id"] == query_label].index)[:4]

# # choose the rest from negative ids
# other_id_idxes = set(gallery_df[gallery_df["id"] != query_label].index)
# other_id_idxes = random.sample(other_id_idxes - set(same_id_idxes), N_GALLERY - len(same_id_idxes))

# gallery_idxes = other_id_idxes + same_id_idxes
# gallery_sample = [(x[1]["path"], x[1]["id"]) for x in gallery_df.loc[gallery_idxes].iterrows()]

# #Create input from gallery images

# gallery_images = [Image.open(DATA_ROOT + x) for x, _ in gallery_sample]
# gallery_labels = [y for _, y in gallery_sample]
# X_gallery = torch.stack(tuple(map(data_transforms, gallery_images))).to(device)
# # print(X_gallery.shape)

# def fliplr(img):
#     """flip images horizontally in a batch"""
#     inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
#     inv_idx = inv_idx.to(img.device)
#     img_flip = img.index_select(3, inv_idx)
#     return img_flip


# def extract_feature(model, X, device="cuda"):
#     """Exract the embeddings of a single image tensor X"""
#     if len(X.shape) == 3:
#         X = torch.unsqueeze(X, 0)
#     X = X.to(device)
#     feature = model(X).reshape(-1)

#     X = fliplr(X)
#     flipped_feature = model(X).reshape(-1)
#     feature += flipped_feature

#     fnorm = torch.norm(feature, p=2)
#     return feature.div(fnorm)


# def get_scores(query_feature, gallery_features):
#     """Calculate the similarity scores of the query and gallery features"""
#     query = query_feature.view(-1, 1)
#     score = torch.mm(gallery_features, query)
#     score = score.squeeze(1).cpu()
#     score = score.numpy()
#     return score



# f_query = extract_feature(model, X_query).detach().cpu()
# f_gallery = [extract_feature(model, X) for X in X_gallery]
# f_gallery = torch.stack(f_gallery).detach().cpu()
# scores = get_scores(f_query, f_gallery)
# print(scores)

# good_trans = transforms.Pad(4, (0, 255, 0)) # green border
# bad_trans = transforms.Pad(4, (255, 0, 0)) # red border

# gallery_images = [img.resize((112, 112)) for img in gallery_images]
# display_images = [(good_trans(img) if lab == query_label else bad_trans(img)) \
#                       for img, lab in zip(gallery_images, gallery_labels)]
# display_images = [display_images[i] for i in np.argsort(scores)[::-1]]

# #Plot gallery images from the highest score to the lowest
# N_ROWS, N_COLS = 4, 8
# score_labels = scores[np.argsort(scores)[::-1]]
# fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 8))


# import matplotlib.pyplot as plt
# N_ROWS, N_COLS = 4, 8
# score_labels = scores[np.argsort(scores)[::-1]]

# fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 8))
# for i in range(N_GALLERY):
#     axes[i // N_COLS, i % N_COLS].imshow(display_images[i])
# for i, ax in enumerate(axes.flat):
#     ax.set_xticks([])
#     ax.set_xticks([], minor=True)
#     ax.set_yticks([])
#     ax.set_yticks([], minor=True)
#     ax.set_xlabel(str(round(score_labels[i], 3)))
#     for spine in [ax.spines.left, ax.spines.right, ax.spines.top, ax.spines.bottom]:
#         spine.set(visible=False)
# plt.show()

# # if out of memory issues arise, we come here to cleanup
# import gc
# gc.collect()
# torch.cuda.empty_cache()
