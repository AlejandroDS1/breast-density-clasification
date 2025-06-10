import argparse
import sys

import torch
import logging
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
    CheckpointLoader,
    ClassificationSaver,
)
from monai.handlers.utils import from_engine
from monai.inferers import SimpleInferer
from monai.networks.nets import densenet121

print_config()

from utils import get_transform

def main():
    parser = argparse.ArgumentParser(description="Train CNN with MONAI") # TODO Se puede hacer un manual mas exhaustivo
    parser.add_argument('--csv_file', type=str, default='./test.csv', help='Path to the CSV file')
    parser.add_argument('--folder_path', type=str, default='./test', help='Path to the image folder')
    parser.add_argument('--dataset', type=str, default='CBIS', help='What dataset is being trained with. Possible values: ["CBIS", "RSNA"]')
    parser.add_argument('--filename', type=str, default='predictions_softmax.csv', help='Filename of the predictions file ')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers (num threads)')
    parser.add_argument('--img_size', nargs='+', type=int, help='Image resolution')
    parser.add_argument('--activation', type=str, default='softmax', help='Last neuron function activation. (prob, discrete, softmax)')
    parser.add_argument('--model_path', type=str, default='./model/checkpoint.pt', help='model file path, checkpoint.pt')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='GPU to use')

    args = parser.parse_args()

    activation = args.activation.lower()
    if activation not in {'prob', 'discrete', 'softmax'}:
        raise ValueError(f"Activation function {args.activation} is not correct.")
    
    torch.multiprocessing.set_sharing_strategy('file_system')    

    ##################################################################################################################
    # Creamos el datalist: Diccionario con el formato necesario para que monai interprete los datos.

    # test = [{"training" : [{ "image" : "(image path)", "label" : int(breast_density | label)}, {...}, {...}]}]
    ##################################################################################################################

    if args.dataset.upper() == 'CBIS': from utils import create_datalist_CBIS as create_datalist
    elif args.dataset.upper() == 'RSNA': from utils import create_datalist_RSNA as create_datalist
    else: raise ValueError(f"{args.dataset} dataset is not supported.")

    test_datalist = create_datalist(args.folder_path, args.csv_file, extension='.png', test=True)

    # Este transform aplica modificaciones aleatorias para evitar que la red neuronal se aprenda tan rápido las imágenes
    # Esto pretende mejorar la generalicación, mejorando el rendimiento de la inferencia

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
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=4).to(device)

    # Logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Metrica clave
    key_val_metric = { 
        "val_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"])),
    }

    # Precision adicional
    extra_metrics = {
            "val_prec": ignite.metrics.Precision(from_engine(["pred", "label"]))
    }

    # Activataion function
    if activation == 'prob':
        act = None
    elif activation == 'discrete':
        act = AsDiscreted(keys='pred', argmax=True)
        key_val_metric = None
    elif activation == 'softmax': # Valor por defecto
        act = Activationsd(keys="pred", softmax=True)


    # Evaluator
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        inferer=SimpleInferer(),
        key_val_metric=key_val_metric,
        additional_metrics=extra_metrics,
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