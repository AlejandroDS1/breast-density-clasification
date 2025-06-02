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
    parser.add_argument('--filename', type=str, default='predictions_softmax.csv', help='Filename of the predictions file ')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers (num threads)')
    parser.add_argument('--img_size', nargs='+', type=int, help='Image resolution')
    parser.add_argument('--binary', action='store_true', help='If there are two clases or only 1')
    parser.add_argument('--activation', type=str, default='softmax', help='Last neuron function activation. (prob, discrete, softmax)')
    parser.add_argument('--model_path', type=str, default='./model/checkpoint.pt', help='model file path, checkpoint.pt')
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

    transform = get_transform(train=False, img_size = args.img_size)

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
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=2 if args.binary else 4).to(device)

    # Logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Activataion function

    key_val_metric = { 
                       "val_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"])),
                       "val_prec": ignite.metrics.Precision(from_engine(["pred", "label"]))
                      }

    if activation == 'prob':
        act = None
    elif activation == 'discrete':
        act = AsDiscreted(keys='pred', argmax=True)
        key_val_metric = None
    elif activation == 'softmax':
        act = Activationsd(keys="pred", softmax=True)

    # Evaluator (Validation)
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        inferer=SimpleInferer(),
        key_val_metric=key_val_metric,
        postprocessing=act,
        val_handlers=[
            StatsHandler(),
            CheckpointLoader(load_path=args.model_path, load_dict={"model": model}, map_location=device),
            ClassificationSaver(
                    filename=args.filename,
                    batch_transform=lambda batch: batch[0]["image"].meta,
                    output_transform=from_engine(["pred"])
            ),
        ],
    )

    evaluator.run()

if __name__ == '__main__':
    main()
