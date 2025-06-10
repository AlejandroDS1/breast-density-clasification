import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

path = "./train.csv"
df = pd.read_csv(path)

# En el caso de RSNA
df.dropna(subset=['density'], inplace=True)
df = df.loc[(df['machine_id'] == 0) & (df['imlpant'] == 0)]


def remove_label_by_area_comparison(image):
    # Umbral binario: segmenta regiones blancas
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    # Encuentra contornos externos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar por área descendente
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)

    for contour in sorted_contours[:-1]: # Nos quedamos solo con el objeto más grande, el pecho.

        # Extraer bbox y rellenar con negro
        x, y, w, h = cv2.boundingRect(contour)
        image[y:y + h, x:x + w] = 0

    return image


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

    max_value = max(sumL, sumR)

    if max_value == sumR:
        img = np.flip(img, axis=1) # Inverir la imagen
    elif max_value == sumT:
        img = np.rot90(img, k=1, axes=(0,1)) # Rotar en sentido antihorario
    elif max_value == sumB:
        img = np.rot90(img, k=1, axes=(1,0)) # Rotar en sentido horario

    return img

def crop_resize_image(row, input_dir, output_dir, img_size=(512, 512), min_area_threshold=5000, margin=10):
    filename = f"{int(row['patient_id'])}_{int(row['image_id'])}.png"
    input_path = Path(input_dir) / filename
    output_path = Path(output_dir) / filename

    if not input_path.exists():
        print(f"[SKIP] No encontrado: {filename}")
        return

    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE) # Leer la imagen en escala de grises

    image = remove_label_by_area_comparison(image) # Eliminar la etiqueta, normalmente CC o MLO

    image = preprocess_directional_threshold(image) # Orientar la imagen

    if image is None:
        print(f"[SKIP] No se pudo leer: {filename}")
        return

    # Procesamiento de contornos
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    if not large_contours:
        print(f"[SKIP] Sin contornos grandes: {filename}")
        return

    # Bounding box combinada de todos los contornos grandes
    x_min, y_min, x_max, y_max = image.shape[1], image.shape[0], 0, 0
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    # Aplicar margen
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, image.shape[1])
    y_max = min(y_max + margin, image.shape[0])

    # Crop y resize directo
    cropped = image[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, img_size, interpolation=cv2.INTER_CUBIC)

    # Guardar resultado
    cv2.imwrite(str(output_path), resized)
    print(f"[OK] Guardado: {filename}")


def crop_and_resize_mammograms_parallel(df, input_dir, output_dir, img_size=(512, 512), min_area_threshold=5000, margin=10, max_workers=8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Procesando imágenes con resize directo a", img_size)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            executor.submit(
                crop_resize_image,
                row, input_dir, output_dir,
                img_size, min_area_threshold, margin
            )

crop_and_resize_mammograms_parallel(
    df,
    input_dir=path,
    output_dir="./test/",
    img_size=(224, 224),
    margin=10,
    max_workers=8
)