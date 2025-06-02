import argparse
import os
import sys
import logging
import torch
import pandas as pd
import cv2
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler, 
    CheckpointSaver,
    ValidationHandler
)
from monai.inferers import SimpleInferer
from monai.networks.nets import densenet121
from monai.handlers.utils import from_engine
import ignite.metrics

from utils import create_datalist, get_transform, create_info_file

def main():
    parser = argparse.ArgumentParser(description="Train CNN with MONAI")
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--csv_file', type=str, default='/home/acantero/tfg/datalists/train_all.csv', help='Path to the CSV file')
    parser.add_argument('--folder_path', type=str, default='/home/acantero/tfg/dataset/train_processed/', help='Path to the image folder')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Batch size for training')
    parser.add_argument('--img_size', nargs='+', type=int, help='Image resolution')
    parser.add_argument('--train_size', type=float, default=1.0, help='Training data proportion')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation data proportion')
    parser.add_argument('--binary', action='store_true', help='If there are two clases or only 1')
    parser.add_argument('--out_dir', type=str, default='./model', help='Output directory')
    parser.add_argument('--cuda', type=str, default='cuda:1', help='GPU to use')
    parser.add_argument('--n_data', type=int, default=-1, help='GPU to use')

    args = parser.parse_args()

    # Coomprovaciones de argumentos antes de empezar

    if args.n_data == 0 or args.n_data < -1:
        raise ValueError("n_data must be greater than 0")


    if 'TITAN' in torch.cuda.get_device_name(args.cuda):
        raise ValueError("TITAN GPU detected, please use a different GPU")

    print_config()
    
    create_info_file(args)
    
    train_datalist = create_datalist(
        path=args.folder_path,
        csv_path=args.csv_file,
        train_size=args.train_size,
        val_size=args.val_size
    )

    train_dict = train_datalist['training']
    val_dict = train_datalist['validation']

    if args.n_data > 1:
        train_dict = train_dict[:args.n_data]
        val_dict = val_dict[:args.n_data]

    transform = get_transform(img_size = args.img_size) 
    

    dataset = Dataset(data=train_dict, transform=transform)
    valdata = Dataset(data=val_dict, transform=transform)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")

    csv = pd.read_csv(args.csv_file)
    values = csv["breast_density"].value_counts().sort_index().values
    class_weights = 1.0 / values
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = densenet121(spatial_dims=2, in_channels=1, out_channels=2 if args.binary else 4, pretrained=True).to(device)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-4) 
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)

    cv2.setNumThreads(0)

    if device.type == 'cpu':
        pin_memory, persistent_workers = True, False
    else:
        pin_memory, persistent_workers = False, False

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        inferer=SimpleInferer(),
        key_val_metric={
            "val_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"])),
        },
        additional_metrics={
            "val_prec": ignite.metrics.Precision(from_engine(["pred", "label"]))
        },
        val_handlers=[StatsHandler(iteration_log=True), TensorBoardStatsHandler(iteration_log=True)],
    )

    save_interval = max(int(args.epochs / 4), 1)

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.epochs,
        train_data_loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=False),
        network=model,
        optimizer=optimizer,
        loss_function=loss_func,
        inferer=SimpleInferer(),
        key_train_metric={"val_acc": ignite.metrics.Accuracy(from_engine(["pred", "label"]))},
        train_handlers=[
            ValidationHandler(validator=evaluator, epoch_level=True, interval=1),
            CheckpointSaver(
                save_dir=args.out_dir,
                save_dict={"model": model,
                           "optimizer": optimizer,
                           "loss_func": loss_func 
                            },
                save_interval=save_interval,
                save_final=True,
                save_key_metric=True,
                key_metric_name='val_acc',
                key_metric_n_saved=3,
                final_filename="checkpoint.pt",
            ),
            StatsHandler(),
            TensorBoardStatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        ],
    )

    trainer.run()


if __name__ == "__main__":
    main()
