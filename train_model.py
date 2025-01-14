from ultralytics import YOLO

# Ruta al archivo de configuración YAML y al modelo 
data_yaml_path = "C:/Users/Xavi/Desktop/Examen/vehicle_recognition/datasets/vehicle_dataset/Vehicle_Detection_Image_Dataset/data.yaml"
pretrained_model = "yolov8n.pt"  # Modelo base 

# Inicializar el modelo YOLO
model = YOLO(pretrained_model)

# Entrenamiento del modelo
model.train(
    data=data_yaml_path,      # Ruta al archivo data.yaml
    epochs=50,                # Número de épocas de entrenamiento
    imgsz=640,                # Tamaño de las imágenes
    batch=16,                 # Tamaño del batch
    name="vehicle_detection", # Nombre del experimento
    project="C:/Users/Xavi/Desktop/Examen/vehicle_recognition/models",  # Carpeta donde se guardará el modelo
    device=0                  # Usar GPU (0) o CPU (-1)
)

print("Entrenamiento completado.")
