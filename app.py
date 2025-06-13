from flask import Flask, request, jsonify
import os
import json
import mediapipe as mp
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageEnhance
import io
import base64
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from dotenv import load_dotenv
from fer import FER

app = Flask(__name__)
CORS(app)

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configura las credenciales de Google Drive desde la variable de entorno
CLIENT_SECRET_JSON = os.getenv('GOOGLE_DRIVE_CREDENTIALS')
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1v8Xss5sKEEgyPHfEBtXYBTHtUevdrhjd'

# Traducción de emociones
TRADUCCION_EMOCIONES = {
    "angry": "enojado",
    "disgust": "disgustado",
    "fear": "miedo",
    "happy": "feliz",
    "sad": "triste",
    "surprise": "sorprendido",
    "neutral": "neutral"
}


def obtener_servicio_drive():
    """Inicializa el servicio de Google Drive."""
    try:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(CLIENT_SECRET_JSON), scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        raise Exception(f"Error al cargar las credenciales: {e}")


def convertir_a_base64(imagen):
    """Convierte una imagen PIL a Base64."""
    buffered = io.BytesIO()
    imagen.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def procesar_imagen_con_puntos(image_np):
    """Procesa la imagen y añade puntos faciales usando Mediapipe."""
    imagen = Image.fromarray(image_np)
    mp_face_mesh = mp.solutions.face_mesh
    puntos_deseados = [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_np)
        if results.multi_face_landmarks:
            draw = ImageDraw.Draw(imagen)
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in puntos_deseados:
                        h, w, _ = image_np.shape
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
                        draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)
    return imagen


@app.route('/upload', methods=['POST'])
def detectar_puntos_y_procesar_imagenes():
    """Procesa la imagen, detecta puntos faciales y emociones."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'})

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'})

    try:
        # Leer y procesar la imagen original
        imagen_pil = Image.open(archivo).convert('RGB')
        imagen_pil = imagen_pil.resize((300, 300))  # Reducir resolución para ahorrar recursos
        imagen_np = np.array(imagen_pil)

        # Mejorar la imagen (contraste y nitidez)
        imagen_mejorada = ImageEnhance.Contrast(imagen_pil).enhance(1.5)
        imagen_mejorada = ImageEnhance.Sharpness(imagen_mejorada).enhance(2.0)

        # Detectar puntos faciales
        imagen_con_puntos = procesar_imagen_con_puntos(imagen_np)

        # Detectar emociones usando FER
        detector = FER(mtcnn=False)  # Desactivar MTCNN para reducir el consumo de recursos
        emociones = detector.detect_emotions(np.array(imagen_mejorada))
        if emociones:
            emocion_principal_en = max(emociones[0]["emotions"], key=emociones[0]["emotions"].get)
            emocion_principal = TRADUCCION_EMOCIONES.get(emocion_principal_en, emocion_principal_en)
        else:
            emocion_principal = "No se detectaron emociones"

        # Subir imagen a Google Drive
        service = obtener_servicio_drive()
        buffered = io.BytesIO()
        imagen_pil.save(buffered, format="PNG")
        archivo_drive = MediaIoBaseUpload(buffered, mimetype='image/png')
        archivo_metadata = {
            'name': archivo.filename,
            'mimeType': 'image/png',
            'parents': [FOLDER_ID]
        }
        archivo_drive_subido = service.files().create(body=archivo_metadata, media_body=archivo_drive).execute()

        # Convertir la imagen procesada a Base64
        img_data_puntos = convertir_a_base64(imagen_con_puntos)

        return jsonify({
            'image_with_points_base64': img_data_puntos,
            'dominant_emotion': emocion_principal,
            'drive_id': archivo_drive_subido.get('id')
        })

    except Exception as e:
        return jsonify({'error': f"Error al procesar la imagen: {str(e)}"})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
