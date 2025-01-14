import cv2
from ultralytics import YOLO

# Rutas de entrada y salida
input_video_path = "datasets/vehicle_dataset/Vehicle_Detection_Image_Dataset/input_video.mp4"
output_video_path = "outputs/output_video.mp4"

# Cargar el modelo YOLOv8 preentrenado
model_path = "models/yolov8n.pt"
model = YOLO(model_path)

# Leer el video de entrada
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error al abrir el video: {input_video_path}")
    exit()

# Configurar el video de salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar predicciones
    results = model.predict(source=frame, save=False, conf=0.5)  # Puedes ajustar el umbral de confianza (conf)
    detections = results[0].boxes.data.cpu().numpy()  # Extraer detecciones

    # Dibujar las detecciones en el frame
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = f"{model.names[int(cls)]} ({conf:.2f})"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Escribir el frame en el video de salida
    out.write(frame)

# Liberar recursos
cap.release()
out.release()
print(f"Video procesado y guardado en: {output_video_path}")
