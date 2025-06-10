import argparse
import sys
import logging
import torch
import cv2
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler, 
    CheckpointSaver,
    ValidationHandler,
)
from monai.inferers import SimpleInferer
from monai.networks.nets import densenet121
from monai.handlers.utils import from_engine
import ignite.metrics

from utils import get_transform, create_info_file, get_loss_func_weights


def main():
    parser = argparse.ArgumentParser(description="Train CNN with MONAI")
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--csv_file', type=str, default='./train.csv', help='Path to the CSV file')
    parser.add_argument('--folder_path', type=str, default='./train/', help='Path to the training image folder.')
    parser.add_argument('--dataset', type=str, default='CBIS', help='What dataset is being trained with. Possible values: ["CBIS", "RSNA"]')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of threadss.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Drop out probability to pass to DenseNet.')
    parser.add_argument('--img_size', nargs='+', type=int, help='Image resolution.')
    parser.add_argument('--train_size', type=float, default=1.0, help='Training data proportion.')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation data proportion.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate to the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay to Adam optimizer.')
    parser.add_argument('--out_dir', type=str, default='./model', help='Output directory.')
    parser.add_argument('--cuda', type=str, default='cuda:0', help='GPU to use.')
    parser.add_argument('--n_data', type=int, default=-1, help='If you want to use limited ammount of data.')

    args = parser.parse_args()

    # Coomprovaciones de argumentos antes de empezar
    if args.n_data == 0 or args.n_data < -1:
        raise ValueError("n_data must be greater than 0")

    print_config()
    
    create_info_file(args)
    
    # Obtencion del datalist con sus labels
    if args.dataset.upper() == 'CBIS': from utils import create_datalist_CBIS as create_datalist
    elif args.dataset.upper() == 'RSNA': from utils import create_datalist_RSNA as create_datalist
    else: raise ValueError(f"{args.dataset} dataset is not supported.")

    train_datalist = create_datalist(
        path=args.folder_path,
        csv_path=args.csv_file,
        train_size=args.train_size,
        val_size=args.val_size
    )

    train_dict = train_datalist['training']
    val_dict = train_datalist['validation']
    
    print(f"# Train: {len(train_dict)}\nVal_dict: {len(val_dict)}")

    if args.n_data > 1:
        train_dict = train_dict[:args.n_data]
        val_dict = val_dict[:args.n_data]

    # Obtención de los objetos Compose, que modifican las imágenes con el data augmentation por cada iteración
    train_transform = get_transform(img_size = args.img_size) 
    test_transform = get_transform(img_size = args.img_size, train = False) 
    

    dataset = Dataset(data=train_dict, transform=train_transform)
    valdata = Dataset(data=val_dict, transform=test_transform)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    
    print(f"Using: {device}")
    
    # Calculo de los pesos para la función de pérdida
    class_weights = torch.tensor(get_loss_func_weights(train_dict), dtype=torch.float32).to(device)

    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Definición del modelo, densenet121 con pesos preentrenados por ImageNet
    model = densenet121(spatial_dims=2, in_channels=1, out_channels=4, pretrained=True, dropout_prob = args.dropout).to(device)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Definición del optimizador Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cv2.setNumThreads(0)

    if device.type == 'cpu':
        pin_memory, persistent_workers = True, False
    else:
        pin_memory, persistent_workers = False, False

    save_interval = max(int(args.epochs / 4), 1)
    
    # Definicion del evaluator. Encargado de hacer la validación durante el entrenamiento
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=DataLoader(valdata, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        inferer=SimpleInferer(),
        key_val_metric={
            "val_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"])), # Calculamos métricas de Accuracy durante la validación.
        },
        additional_metrics={
            "val_prec": ignite.metrics.Precision(from_engine(["pred", "label"])) # Calculamos métricas de Precision durante la validación
        },
        val_handlers=[
            StatsHandler(iteration_log=True),
            TensorBoardStatsHandler(iteration_log=True),
            CheckpointSaver(
                save_dir=args.out_dir,
                save_dict={"model": model,
                           "optimizer": optimizer,
                           "loss_func": loss_func 
                            },
                save_interval=save_interval,
                save_final=True,
                save_key_metric=True,
                key_metric_name='val_acc', # Se crean 3 archivos con la mejor validación obtenida.
                key_metric_n_saved=3,
                final_filename="checkpoint.pt",
            ),
        ],
    )

    # Definicion del trainer. Encargado de gestionar todo el proceso de entrenamiento
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        optimizer=optimizer,
        loss_function=loss_func,
        inferer=SimpleInferer(),
        key_train_metric={"train_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"]))}, # Calculo de la Accuracy en entrenamiento. 
        train_handlers=[
            ValidationHandler(validator=evaluator, epoch_level=True, interval=1),
            StatsHandler(),
            TensorBoardStatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)), # Obtenemos las métricas de loss. Valores de salida de la función de pérdida.
        ],
    )

    trainer.run() # Empieza el entrenamiento del modelo.


if __name__ == "__main__":
    main()