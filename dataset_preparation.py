import os
from kagglehub import dataset_download

def prepare_dataset():
    print("Iniciando la descarga del dataset desde Kaggle...")
    path = dataset_download("farzadnekouei/top-view-vehicle-detection-image-dataset")
    print(f"Path al dataset descargado: {path}")
    
    # Crear estructura en el directorio local
    local_path = "datasets/vehicle_dataset/Vehicle_Detection_Image_Dataset"
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
    
    # Mover el dataset descargado a la carpeta correspondiente
    os.system(f'xcopy /E /Y "{path}" "{local_path}"')
    print(f"Dataset preparado en: {local_path}")

if __name__ == "__main__":
    prepare_dataset()
