import torch
import pandas as pd
import subprocess
from load_model import load_model_from_opts


VeRi = "/home/toms.zinars/Anzelika/Otrais_datasets_divas_katrs_katraa/"


train_csv_path = VeRi + "train.csv"
val_csv_path = VeRi + "val.csv"


model = load_model_from_opts("/home/toms.zinars/Anzelika/veri+vehixlex_unmodified/opts.yaml", 
                             ckpt="/home/toms.zinars/Anzelika/veri+vehixlex_unmodified/net_39.pth", 
                             remove_classifier=True)


def continue_training(model, train_csv_path, val_csv_path, save_dir, total_epoch=20, warm_epoch=3, batchsize=16, save_freq=1, fp16=True, erasing_p=0.5):
    command = f"python3 /home/toms.zinars/Anzelika/vehicle_reid_repo2/vehicle_reid/train.py --data_dir='/home/toms.zinars/Anzelika/Otrais_datasets_divas_katrs_katraa/' \
        --name='{save_dir}' \
        --train_csv_path='{train_csv_path}' \
        --val_csv_path='{val_csv_path}' \
        --batchsize={batchsize} \
        --total_epoch={total_epoch} \
        --warm_epoch={warm_epoch} \
        --save_freq={save_freq} \
        {'--fp16' if fp16 else ''} \
        --erasing_p={erasing_p} \
        --checkpoint='/home/toms.zinars/Anzelika/veri+vehixlex_unmodified/net_39.pth'"
    
    subprocess.run(command, shell=True)

save_dir = '/home/toms.zinars/Anzelika/Svari_Tomass/'

continue_training(model, train_csv_path, val_csv_path, save_dir)


torch.save(model.state_dict(), "/home/toms.zinars/Anzelika/Svari_Tomass/Tomass_upgrade.pth")
