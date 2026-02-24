import os
import sys
import cv2
import whisper
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Configuración OpenAI vía Variable de Entorno (o pega tu clave aquí)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

# --- CARGA DE MEDIAPIPE CON SEGURIDAD ---
mp_face_mesh = None
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    print("MediaPipe cargado exitosamente.")
except Exception as e:
    print(f"Aviso: MediaPipe no disponible en este servidor. Error: {e}")

# --- CARGA DE WHISPER ---
print("Cargando Whisper...")
model_stt = whisper.load_model("base")

# Inicializar motor de rostro SOLO si MediaPipe cargó bien
face_mesh_engine = None
if mp_face_mesh:
    try:
        face_mesh_engine = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    except Exception as e:
        print(f"No se pudo iniciar el motor de rostros: {e}")
        face_mesh_engine = None

def analizar_video(path):
    # --- ANÁLISIS NO VERBAL (Solo si el motor funciona) ---
    score_nv = 0.0
    if face_mesh_engine:
        try:
            cap = cv2.VideoCapture(path)
            frames_con_rostro = 0
            total_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                total_frames += 1
                if total_frames % 15 == 0: # Saltamos más frames para no saturar el servidor
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh_engine.process(rgb)
                    if results.multi_face_landmarks:
                        frames_con_rostro += 1
            cap.release()
            score_nv = round(min(5, (frames_con_rostro / (total_frames / 15)) * 5), 1) if total_frames > 0 else 0
        except:
            score_nv = 0.0

    # --- ANÁLISIS VERBAL (Siempre funciona) ---
    print("Transcribiendo...")
    texto = model_stt.transcribe(path, language="es")["text"]
    
    print("Evaluando calidez...")
    prompt = f"Evalúa calidez (IECCT-SV El Salvador) en JSON: {texto}"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    return {
        "verbal": json.loads(response.choices[0].message.content),
        "no_verbal": score_nv,
        "texto": texto
    }

@app.route('/auditar', methods=['POST'])
def auditar():
    if 'video' not in request.files:
        return jsonify({"error": "No hay video"}), 400
    file = request.files['video']
    temp_path = "video_temp.mp4"
    file.save(temp_path)
    try:
        return jsonify(analizar_video(temp_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

if __name__ == '__main__':
    # Usar puerto de Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
