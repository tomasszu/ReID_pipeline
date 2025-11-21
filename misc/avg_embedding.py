import numpy as np
import os
import torch

embeddings_path = "/home/tomass/tomass/ReID_pipele/embeddings/AI_City_Images/cam4"

def load_text_embeddings(list):
    embeddings = []
    for file in list:
        embedding = torch.load(f'{embeddings_path}/{file}').cpu().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)



if __name__ == "__main__":

    dir_list = os.listdir(embeddings_path)    

    embeddings = load_text_embeddings(dir_list)
    print(f'Loaded {len(embeddings)} text embeddings with shape: {embeddings[0].shape}')
    

    average_embedding = torch.mean(torch.tensor(embeddings), dim=0)
    torch.save(average_embedding, f'/home/tomass/tomass/ReID_pipele/embeddings/AI_City_Images/averaged/average_cam4.pt')