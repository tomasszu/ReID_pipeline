import numpy as np
import os
import sys
import torch
from PIL import Image
from torchvision import transforms


sys.path.append('/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2')

from vehicle_reid.load_model import load_model_from_opts

####### COMUNICATION FUNCTIONS ##########

# def image_embedding_extract(query: str, address: str = 'http://10.13.135.161:5000/query'):
#     """Call HPC server to return text embedding"""
#     files = {
#         "query": ("query", query, "text/plain"),
#     }
#     response = requests.post(address, files=files)
#     txt_embed = np.frombuffer(response.content, dtype=np.float64)

#     print(txt_embed.shape) # you can remove this
#     return txt_embed

def image_batch_embedding_extract(path, batch, output_path, model, data_transforms, device):
    if len(batch) > 0:
        image_batch = []
        for file in batch:
            filepath = path + "/" + file
            img = Image.open(filepath)
            image_batch.append(img)

        X_images = torch.stack(tuple(map(data_transforms, image_batch))).to(device)
        features = extract_batch_features(model, X_images)
        for feature, file in zip(features, batch):
            torch.save(feature.clone().detach(), f'{output_path}/{file}.pt') 
        # print("[DEBUG] extracted batch features: \n")
        # print(features.shape)
    else: return




def extract_batch_features(model, X):
    """Extract features for a batch of images (X: B x C x H x W)"""
    X = X.to(device)
    features = model(X)  # shape: (B, D)

    X_flipped = fliplr(X)
    flipped_features = model(X_flipped)  # shape: (B, D)

    features += flipped_features

    fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
    normalized = features.div(fnorm)
    return normalized  # shape: (B, D)

    

def fliplr(img):
    """flip images horizontally in a batch"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip



if __name__ == "__main__":

    input_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/dataset"
    output_path = "embeddings/AI_City_Images/cam4"
    dir_list = os.listdir(input_path)
    dir_list = dir_list[:10]
    print(f"[File import] Read in {len(dir_list)} files.")

    batchsize = 16
    it = 0

    device = "cuda"
    model = load_model_from_opts("vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", 
                                    ckpt="vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", 
                                    remove_classifier=True)
    model.eval()
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        if (batchsize * (it+1)) < len(dir_list):
            batch = dir_list[batchsize * it: (batchsize * (it+1))] # otraja pozicija noradits skaitlis LIDZ kuram (neieskaitot) ņem elementus
            print(f"\nLen:[{batchsize * it}:{(batchsize * (it+1))-1}], actual count = {len(batch)}")
            image_batch_embedding_extract(input_path,batch,output_path, model, data_transforms, device)
            print(batch)
            print("\n---------------")            
        else:
            batch = dir_list[batchsize * it:]
            print(f"\nLen:[{batchsize * it}:{len(dir_list)}], actual count = {len(batch)}")
            image_batch_embedding_extract(input_path,batch,output_path, model, data_transforms, device)
            print(batch)
            print("\n------END------")
            break
        it += 1


    # for file in dir_list[:10]:
    #     print(path + "/" + file)
    #     # text_embed = text_embedding_request(query, address='http://10.13.135.161:5000/query')
    #     # torch.save(torch.tensor(text_embed), f'txt_embed/construction_embeddings/{query}_embedding.pt')
    #     # print(f'Saved text embedding for "{query}"')