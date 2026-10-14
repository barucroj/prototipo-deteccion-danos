import cv2
import numpy as np


def apply_clahe_rgb(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica el algoritmo CLAHE sobre el canal de Luminancia (L) en el espacio de color LAB
    para mejorar el contraste de rayaduras y abolladuras sin alterar los colores.
    
    Args:
        image (np.ndarray): Imagen en formato BGR (leída comúnmente por OpenCV).
        clip_limit (float): Límite de contraste para el recorte de histograma.
        tile_grid_size (tuple): Tamaño de la rejilla para el histograma local.
        
    Returns:
        np.ndarray: Imagen con contraste mejorado en formato BGR.
    """
    # Convertir de BGR a LAB para separar la luminancia del color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Crear el objeto CLAHE y aplicarlo al canal L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)
    
    # Fusionar los canales modificados y regresar al espacio BGR
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def resize_with_padding(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Redimensiona una imagen manteniendo la relación de aspecto original
    y añade padding negro para alcanzar el tamaño objetivo exacto.
    """
    h, w = image.shape[:2]
    th, tw = target_size
    
    # Calcular factor de escala manteniendo relación de aspecto
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Crear un lienzo negro del tamaño objetivo
    padded_image = np.zeros((th, tw, 3), dtype=np.uint8)
    
    # Calcular coordenadas para centrar la imagen original escalada en el lienzo
    x_offset = (tw - new_w) // 2
    y_offset = (th - new_h) // 2
    
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded_image


def to_pytorch_tensor_format(image: np.ndarray, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Normaliza los píxeles al rango [0, 1], aplica la normalización estadística
    estándar (Media y Desviación Estándar) y transpone las dimensiones al formato
    requerido por PyTorch: (Canales, Alto, Ancho) -> CHW.
    
    Nota: Transforma de BGR a RGB internamente.
    """
    # Cambiar espacio de color de BGR (OpenCV) a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Escalar rango a [0.0, 1.0]
    normalized = image_rgb.astype(np.float32) / 255.0
    
    # Aplicar Z-score normalización usando arreglos de NumPy
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Cambiar dimensiones de HWC (Alto, Ancho, Canales) a CHW (Canales, Alto, Ancho)
    # Requisito indispensable para cargarlo en tensores de PyTorch
    tensor_format = np.transpose(normalized, (2, 0, 1))
    return tensor_format


def preprocess_pipeline(image_path: str, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Pipeline unificado de preprocesamiento para inferencia o entrenamiento.
    """
    # 1. Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde: {image_path}")
        
    # 2. Aplicar mejora de contraste adaptativo (CLAHE)
    img_contrast = apply_clahe_rgb(img)
    
    # 3. Redimensionar manteniendo aspecto y aplicando padding
    img_resized = resize_with_padding(img_contrast, target_size=target_size)
    
    # 4. Formatear y normalizar estadísticamente para PyTorch
    img_tensor = to_pytorch_tensor_format(img_resized)
    
    return img_tensor