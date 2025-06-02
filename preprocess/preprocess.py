import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_directional(img: np.ndarray) -> np.ndarray:
    """
    Asegura que el pecho siempre provenga de ARRIBA.
    Puede invertir horizontal y/o rotar verticalmente si es necesario.
    """
    y, x = img.shape

    # Comparación horizontal
    sumL, sumR = np.sum(img[:, :x//2]), np.sum(img[:, x//2:])
    # Comparación vertical
    sumT, sumB = np.sum(img[:y//2, :]), np.sum(img[y//2:, :])

    # Si el pecho viene de la derecha, voltear horizontalmente
    if sumR > sumL:
        img = np.flip(img, axis=1)

    # Si el pecho viene de abajo, rotar 180 grados
    if sumB > sumT:
        img = np.flip(img, axis=0)

    return img

import glob

def process_directory(img_dir, output_dir=None, show=False):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    paths = glob.glob(os.path.join(img_dir, "*.png"))  # puedes usar *.jpg si aplica
    
    for img_path in paths:
        img = np.array(Image.open(img_path).convert('L'))  # blanco y negro

        processed = preprocess_directional(img)

        if output_dir:
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            Image.fromarray(processed).save(out_path)

        if show:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title("Original")
            plt.imshow(img, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Procesado")
            plt.imshow(processed, cmap='gray')
            plt.axis('off')

            plt.show()


process_directory(img_dir='/home/acantero/tfg/dataset/Train', output_dir='/home/acantero/tfg/dataset/train_processed', show=False)
