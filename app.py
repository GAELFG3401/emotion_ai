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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from flask import send_file

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
def detectar_puntos_y_generar_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió correctamente la imagen'}), 400

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se cargó ninguna imagen'}), 400

    try:
        # 1) Abrir y preprocesar imagen
        imagen_pil = Image.open(archivo).convert('RGB').resize((300, 300))
        imagen_np = np.array(imagen_pil)
        imagen_mejorada = ImageEnhance.Contrast(imagen_pil).enhance(1.5)
        imagen_mejorada = ImageEnhance.Sharpness(imagen_mejorada).enhance(2.0)

        # 2) Dibujar puntos faciales
        imagen_con_puntos = procesar_imagen_con_puntos(imagen_np)

        # 3) Detectar emoción
        detector = FER(mtcnn=False)
        emociones = detector.detect_emotions(np.array(imagen_mejorada))
        if emociones:
            emo_en = max(emociones[0]["emotions"], key=emociones[0]["emotions"].get)
            emocion = TRADUCCION_EMOCIONES.get(emo_en, emo_en)
        else:
            emocion = "No se detectaron emociones"

        # 4) (Opcional) subir la imagen original a Drive...
        #    — tu código actual de obtener_servicio_drive() y subida —

        # 5) Generar PDF en memoria
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter  # 612×792 puntos

        # Dibujar la imagen con puntos en el PDF
        img_io = io.BytesIO()
        imagen_con_puntos.save(img_io, format='PNG')
        img_io.seek(0)
        # Ajusta el tamaño/posición a tu gusto:
        img_width = 300
        img_height = 300
        x_img = (width - img_width) / 2
        y_img = height - img_height - 100
        c.drawImage(ImageReader(img_io), x_img, y_img, img_width, img_height)

        # Escribir el texto de la emoción
        text_x = 50
        text_y = y_img - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(text_x, text_y, f"Emoción detectada: {emocion}")

        c.showPage()
        c.save()
        pdf_buffer.seek(0)

        # 6) Devolver el PDF
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='resultado.pdf'
        )

    except Exception as e:
        return jsonify({'error': f"Error al procesar la imagen: {e}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
