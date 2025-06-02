import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityD,
    ResizeD,
    RandFlipD,
    RandRotateD,
    RandZoomD, 
    RandGaussianNoiseD,
    ToTensor
)

def create_datalist(path, csv_path, extension='.png', train_size=1.0, val_size=0.15, random_state=42, test=False):

    csv = pd.read_csv(csv_path)
    datalist = {"training": [], "validation": [], "testing": []}
    data = []

    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(extension):
                absolute_file_path = os.path.join(dirpath, file)
                try:
                    label = int(csv.loc[csv['image_file_path'] == absolute_file_path[len(path):], 'breast_density'].values[0]) - 1 
                    data.append({'image': absolute_file_path, 'label': label})
                except:
                    print(f"File {absolute_file_path[len(path):]} was unable to use")

    if test:
        return data

    if train_size == 1.0:
        train, val = train_test_split(data, test_size=val_size, random_state=random_state)
    else:
        train_, test_data = train_test_split(data, train_size=train_size, random_state=random_state)
        train, val = train_test_split(train_, test_size=val_size, random_state=random_state)
        datalist['testing'] = test_data

    datalist['training'] = train
    datalist['validation'] = val

    return datalist


def get_transform(train : bool = True, img_size = (1200, 800)):

    compose_list = [
                        LoadImageD(keys="image", image_only=True),
                        EnsureChannelFirstD(keys="image"),
                        ScaleIntensityD(keys="image"),
                   ]

    if img_size != (1,1):
        compose_list.append(ResizeD(keys="image", spatial_size=img_size, mode='bicubic'))


    if train:
        compose_list = compose_list + [
                                    RandFlipD(keys="image", spatial_axis=0, prob=0.2),              # 20% de probabilidad de voltear horizontal
                                    #RandFlipD(keys="image", spatial_axis=1, prob=0.2),             # 20% de probabilidad de voltear vertical
                                    RandRotateD(keys="image", range_x=0.17, prob=0.3),               # Rotación de ±10 grados (0.17 raidanes)
                                    RandZoomD(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.2),  # Zoom in/out aleatorio
                                    RandGaussianNoiseD(keys="image", prob=0.2, mean=0, std=0.05),   # Ruido gaussiano 20% de las veces                   
                                    #ToTensor(),
                                ]


    return Compose(compose_list)


def create_info_file(args):

    # Guardar argumentos
    with open("info.txt", "w") as f:
        f.write("=== Argumentos del script ===\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

        f.write("\n=== Definición del Transform ===\n")

        # Leer el propio archivo
        script_path = os.path.realpath(__file__)
        inside_transform = False

        f.write(f"GPU used -> {torch.cuda.get_device_name(args.cuda)}\n")

        with open(script_path, "r") as script_file:
            for line in script_file:
                # Buscar inicio del transform
                if "get_transform(" in line and not 'if' in line:
                    inside_transform = True
                    f.write(line)
                    continue

                # Si estamos dentro, escribir las líneas

                if inside_transform: 
                    if line.lstrip().startswith("#"):
                        continue
                    f.write(line)
                    # Buscar final
                    if "return" in line:
                        break
