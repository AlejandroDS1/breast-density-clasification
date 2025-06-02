import argparse
import sys

import torch
import logging
import pandas as pd
import ignite

from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Activationsd,
    AsDiscreted,
)

# SUPERVISED TRAINING
from monai.config import print_config
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointLoader,
    ClassificationSaver,
)
from monai.handlers.utils import from_engine
from monai.inferers import SimpleInferer
from monai.networks.nets import densenet121

print_config()

from utils import create_datalist, get_transform

def main():
    parser = argparse.ArgumentParser(description="Train CNN with MONAI") # TODO Se puede hacer un manual mas exhaustivo
    parser.add_argument('--csv_file', type=str, default='/home/acantero/tfg/datalists/all_test.csv', help='Path to the CSV file')
    parser.add_argument('--folder_path', type=str, default='/home/acantero/tfg/dataset/test_processed/', help='Path to the image folder')
    parser.add_argument('--logs_dir', type=str, default='./runs', help='Runs file directory')
    parser.add_argument('--filename', type=str, default='predictions.csv', help='Filename of the predictions file ')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers (num threads)')
    parser.add_argument('--activation', type=str, default='softmax', help='Last neuron function activation. (prob, discrete, softmax)')
    parser.add_argument('--model_path', type=str, default='/saves/model/checkpoint.pt', help='model file path, checkpoint.pt')
    parser.add_argument('--cuda', type=str, default='cuda:1', help='GPU to use')

    args = parser.parse_args()

    if 'TITAN' in torch.cuda.get_device_name(args.cuda):
        raise ValueError("TITAN GPU detected, please use a different GPU")

    activation = args.activation.lower()
    if activation not in {'prob', 'discrete', 'softmax'}:
        raise ValueError(f"Activation function {args.activation} is not correct.")
    
    torch.multiprocessing.set_sharing_strategy('file_system')    
    ##################################################################################################################
    # Creamos el datalist: Diccionario con el formato necesario para que monai interprete los datos.

    # test = [{"training" : [{ "image" : "(image path)", "label" : int(breast_density | label)}, {...}, {...}]}]
    ##################################################################################################################

    test_datalist = create_datalist(args.folder_path, args.csv_file, extension='.png', test=True)


    # Este transofrm aplica modificaciones aleatorias para evitar que la red neuronal se aprenda tan rapido las imagenes
    # Esto pretende mejorar la generalicacion, mejorando el validation

    transform = get_transform()

    dataset = Dataset(data=test_datalist, transform=transform)
    # Training Configurations
    _device = args.cuda if torch.cuda.is_available() else 'cpu'
    print(f"Using: {_device} Name: {torch.cuda.get_device_name(_device) if _device != 'cpu' else _device}")
    device = torch.device(_device)

    # Dataset-related Settings
    if _device == 'cpu':
        pin_memory, persistent_workers = True, False
    else:
        pin_memory, persistent_workers = False, False


    # Model Definition
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=4).to(device)

    # Logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
    from monai.transforms import Compose, EnsureTyped, AsDiscrete
    from monai.handlers import StatsHandler, TensorBoardStatsHandler
    from monai.engines import SupervisedEvaluator
    from ignite.metrics import Accuracy, Precision
    from ignite.contrib.metrics import ROC_AUC
    from torch.utils.data import DataLoader
    

    # Postprocesado para convertir predicciones y etiquetas a formato adecuado
    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        AsDiscrete(keys="pred", argmax=True, to_onehot=4),
        AsDiscrete(keys="label", to_onehot=4),
    ])

    # Métricas para evaluación
    metrics = {
        # Accuracy y precisión macro
        "val_acc_macro": Accuracy(output_transform=from_engine(["pred", "label"])),
        "val_prec_macro": Precision(output_transform=from_engine(["pred", "label"])),

        # ROCAUC macro
        "val_rocauc_macro": ROC_AUC(from_engine(["pred", "label"]), multi_class='ovo'),

        # Accuracy y precisión por clase usando matriz de confusión
        #"val_confmat": ConfusionMatrixMetric(
        #    include_background=False,
        #    metric_name=("accuracy", "precision"),
        #    reduction="none"  # permite registrar por clase
        #),
    }

    # Evaluador modificado
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False
        ),
        network=model,
        inferer=SimpleInferer(),
        postprocessing=Activationsd(keys="pred", softmax=True),
        key_val_metric=metrics,  # no métrica principal
        additional_metrics=metrics,
        val_handlers=[
            StatsHandler(tag_name="val"),
            TensorBoardStatsHandler(log_dir="./runs", tag_name="val")
        ],
    )

    # Llamar al evaluador para ejecutar el test
    evaluator.run()

if __name__ == '__main__':
    main()
