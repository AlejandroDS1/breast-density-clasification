import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt


def binarize(img: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Binariza la imagen: los píxeles con valor > threshold se ponen a 1 (blanco),
    el resto a 0 (negro).
    """
    if img.max() > 1.0:
        img = img / 255.0  # normaliza si está en escala 0-255
    return (img > threshold).astype(np.uint8)


def preprocess_directional_threshold(img: np.ndarray, rect_ratio: float = 0.2) -> np.ndarray:
    """
    Asegura que el pecho siempre provenga de ARRIBA usando una imagen binarizada.
    Evalúa si hay más 'masa blanca' en los bordes y rota/flipa en consecuencia.
    """
    bin_img = binarize(img)
    y, x = bin_img.shape

    # Tamaño del rectángulo de inspección (20% por defecto)
    rx, ry = int(x * rect_ratio), int(y * rect_ratio)

    # Define rectángulos de interés (bordes)
    left = bin_img[:, :rx]
    right = bin_img[:, -rx:]
    top = bin_img[:ry, :]
    bottom = bin_img[-ry:, :]

    # Conteo de píxeles blancos en cada región
    sumL, sumR = np.sum(left), np.sum(right)
    sumT, sumB = np.sum(top), np.sum(bottom)

    max_value = max(sumL, sumR, sumT, sumB)

    if max_value == sumL:
        img = np.rot90(img)
    elif max_value == sumR:
        img = np.rot90(img, axes=(0, 1))  # Rota 90 grados en sentido antihorario
    elif max_value == sumB:
        img = np.flip(img, axis=0)


    # Ajustes de orientación
    if sumR > sumL:
        img = np.flip(img, axis=1)  # flip horizontal
    if sumB > sumT:
        img = np.flip(img, axis=0)  # flip vertical

    return img

def process_directory(img_dir, output_dir=None, show=False):
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    paths = glob.glob(os.path.join(img_dir, "*.png"))  # ajusta extensión si hace falta

    for img_path in paths:
        img = np.array(Image.open(img_path).convert('L')) / 255.0  # escala 0-1
        processed = preprocess_directional_threshold(img)

        if output_dir:
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            Image.fromarray((processed * 255).astype(np.uint8)).save(out_path)

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

#process_directory(img_dir='/home/acantero/tfg/dataset/Train', output_dir='/home/acantero/tfg/dataset/train_processed', show=False)
