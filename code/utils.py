import pandas as pd
import numpy as np
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
)
from collections import Counter


def create_datalist_CBIS(path, csv_path, extension='.png', train_size=1.0, val_size=0.15, random_state=42, test=False):
    """
    Crea el datalist. Una lista de diccionarios que contiene la imagen de entrada y el label correspondiente:
    datalist = {"image": image_path, # string
                "label" label} # int
    """

    csv = pd.read_csv(csv_path)
    datalist = {"training": [], "validation": [], "testing": []}
    data = []

    for dirpath, _, filenames in os.walk(path): # Iteramos por todo el directorio donde se encuentran las imágenes.
        for file in filenames:
            if file.endswith(extension):
                absolute_file_path = os.path.join(dirpath, file)
                try: # Creamos el diccionario y lo añadimos a la lista
                    label = int(csv.loc[csv['image_file_path'] == absolute_file_path[len(path):], 'breast_density'].values[0]) - 1 
                    data.append({'image': absolute_file_path, 'label': label})
                except:
                    print(f"File {absolute_file_path[len(path):]} was unable to use")

    # Separación en train, validation y test.
    # Nota: Se usa una semilla random_state=42. Se puede pasar None para que sea aleatorio.

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


def create_datalist_RSNA(path, csv_path, extension='.png', train_size=1.0, val_size=0.15, random_state=42, test=False):

    BI_RADS = {'A' : 0, 'B': 1, 'C': 2, 'D': 3} # Transforma los valores de densidad del archivo csv a valores númericos
    csv = pd.read_csv(csv_path)
    csv = csv.dropna(subset=['density']) # Eliminamos Nans de la columna density
    datalist = {"training": [], "validation": [], "testing": []}
    data = []
    l = len(extension)

    for dirpath, _, filenames in os.walk(path):
        for img in filenames:
            
            patient_id, image_id = img.split('_')
            
            # El nombre del archivo contiene el patient_id y el image_id.
            label = csv.loc[(csv['patient_id'] == int(patient_id)) & (csv['image_id'] == int(image_id[:-l]))]['density'].values

            if len(label) == 0: continue

            abs_path = os.path.join(path, img)
            
            if label[0] in BI_RADS.keys():
                label = BI_RADS[label[0]]
            else:
                label = label[0]

            data.append({'image': abs_path, 'label': label})
    
    # Separación en train, validation y test.
    # Nota: Se usa una semilla random_state=42. Se puede pasar None para que sea aleatorio.

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


def get_loss_func_weights(d: dict):

    """
    Esta función calcula los pesos que irán a la función de pérdida.
    
    d: dict
    Tiene que tener la key 'label' con los labels correspondientes.
    """

    _ = Counter([v['label'] for v in d])

    values = []

    n_classes = len(_.keys())

    for i in range(n_classes):
        values.append(_.get(i, 0))

    values = np.array(values)

    class_weights = 1.0 / (values / values.sum()) # Calculamos la inversa de la frequencia

    # Normalizamos (recomendable)
    class_weights = class_weights / class_weights.sum() # class_weights /= class_weights.sum()

    return class_weights

def get_transform(train : bool = True, img_size = (800, 350)):

    """
    Retorna un objeto Compose que implementará las transformaciones de los datos necesarias.
    """   

    compose_list = [
                        LoadImageD(keys="image", image_only=True), # Carga las imágenes
                        EnsureChannelFirstD(keys="image"),         # Asegura que la forma del tensor sea correcta 
                        ScaleIntensityD(keys="image"),             # Escala la intensidad a valores [0, 1] 
                   ]

    if img_size != (1,1): # Si se pasa por parámetro (1, 1) no se aplicará resize
        compose_list.append(ResizeD(keys="image", spatial_size=img_size, mode='bicubic')) # Resize con interpolación bicúbica

    if train: # El transform del train aplica data augmentation cuando
        compose_list = compose_list + [
                                    RandFlipD(keys="image", spatial_axis=0, prob=0.5),              # 50% de probabilidad de voltear horizontal
                                    RandFlipD(keys="image", spatial_axis=1, prob=0.5),              # 50% de probabilidad de voltear vertical
                                    RandRotateD(keys="image", range_x=0.35, prob=0.3),              # Rotación de ±20 grados (0.35 raidanes)
                                    RandZoomD(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.3),  # Zoom in/out aleatorio
                                    RandGaussianNoiseD(keys="image", prob=0.3, mean=0.2, std=0.15),   # Ruido gaussiano 30% de las veces                   
                                ]

    return Compose(compose_list)


def get_transform(train : bool = True, img_size = (800, 350)):

    """
    Retorna un objeto Compose que implementará las transformaciones de los datos necesarias.
    """   

    compose_list = [
                        LoadImageD(keys="image", image_only=True), # Carga las imágenes
                        EnsureChannelFirstD(keys="image"),         # Asegura que la forma del tensor sea correcta 
                        ScaleIntensityD(keys="image"),             # Escala la intensidad a valores [0, 1] 
                   ]

    if img_size != (1,1): # Si se pasa por parámetro (1, 1) no se aplicará resize
        compose_list.append(ResizeD(keys="image", spatial_size=img_size, mode='bicubic')) # Resize con interpolación bicúbica

    if train: # El transform del train aplica data augmentation cuando
        compose_list = compose_list + [
                                    RandFlipD(keys="image", spatial_axis=0, prob=0.5),              # 50% de probabilidad de voltear horizontal
                                    RandFlipD(keys="image", spatial_axis=1, prob=0.5),              # 50% de probabilidad de voltear vertical
                                    RandRotateD(keys="image", range_x=0.35, prob=0.3),              # Rotación de ±20 grados (0.35 raidanes)
                                    RandZoomD(keys="image", min_zoom=0.9, max_zoom=1.1, prob=0.3),  # Zoom in/out aleatorio
                                    RandGaussianNoiseD(keys="image", prob=0.3, mean=0.2, std=0.15),   # Ruido gaussiano 30% de las veces                   
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
