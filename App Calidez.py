import os
import sys

# --- BLOQUE DE CARGA MAESTRA ---
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    print("LOGRADO: MediaPipe cargado correctamente.")
except (ImportError, AttributeError):
    print("Aviso: Error de ruta estándar, aplicando parche de carga profunda...")
    try:
        # Forzamos la carga desde la carpeta site-packages directamente
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        print("LOGRADO: Parche de carga profunda exitoso.")
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudo inicializar MediaPipe. Detalle: {e}")
        sys.exit()

# --- IMPORTACIÓN DE LIBRERÍAS RESTANTES ---
import cv2
import whisper
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Tu API Key (Asegúrate de que sea una nueva y válida)
client = OpenAI(api_key="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")

print("Cargando Whisper (esto puede tomar 1-2 minutos)...")
model_stt = whisper.load_model("base")

# Inicializar motor de rostro una sola vez
face_mesh_engine = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ... (El resto de tu función analizar_video y rutas de Flask se mantienen igual)

def analizar_video(path):
    cap = cv2.VideoCapture(path)
    frames_con_rostro = 0
    total_frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        if total_frames % 10 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh_engine.process(rgb)
            if results.multi_face_landmarks:
                frames_con_rostro += 1
    cap.release()
    score_nv = round(min(5, (frames_con_rostro / (total_frames / 10)) * 5), 1) if total_frames > 0 else 0
    
    print("Transcribiendo...")
    texto = model_stt.transcribe(path, language="es")["text"]
    
    print("Consultando GPT...")
    prompt = f"Evalúa calidez (IECCT-SV) en JSON: {texto}"
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
    file = request.files['video']
    file.save("video_proceso.mp4")
    try:
        return jsonify(analizar_video("video_proceso.mp4"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists("video_proceso.mp4"): os.remove("video_proceso.mp4")

if __name__ == '__main__':
    print("\n--- SERVIDOR ACTIVO EN PORT 5001 ---")
    app.run(port=5001)