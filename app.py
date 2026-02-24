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

# Configuración OpenAI (Usa variable de entorno en Render para seguridad)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))

# --- CARGA PROTEGIDA DE MEDIAPIPE ---
face_mesh_engine = None

try:
    import mediapipe as mp
    # Intentamos inicializar el motor solo si la librería existe
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        face_mesh_engine = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("Motor de MediaPipe cargado correctamente.")
    else:
        print("Aviso: MediaPipe no tiene el atributo 'solutions' en este entorno.")
except Exception as e:
    print(f"Aviso: No se pudo cargar MediaPipe (Visión Artificial desactivada). Error: {e}")

# --- CARGA DE WHISPER (Voz a Texto) ---
print("Cargando Whisper... esto puede tardar unos minutos.")
try:
    model_stt = whisper.load_model("base")
    print("Whisper cargado correctamente.")
except Exception as e:
    print(f"Error cargando Whisper: {e}")

def analizar_video(path):
    # --- DOMINIO 2: ANÁLISIS NO VERBAL ---
    # Si el motor falló al cargar, devolvemos un puntaje base (o 0)
    score_nv = 0.0
    
    if face_mesh_engine is not None:
        try:
            cap = cv2.VideoCapture(path)
            frames_con_rostro = 0
            total_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                total_frames += 1
                if total_frames % 20 == 0: # Analizamos pocos frames para ahorrar memoria en Render
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh_engine.process(rgb)
                    if results and results.multi_face_landmarks:
                        frames_con_rostro += 1
            cap.release()
            # Escala 1-5
            if total_frames > 0:
                score_nv = round(min(5, (frames_con_rostro / (total_frames / 20)) * 5), 1)
        except Exception as vis_err:
            print(f"Error en análisis visual: {vis_err}")
            score_nv = 0.0
    else:
        print("Saltando análisis visual (Motor no disponible).")
        score_nv = 0.0

    # --- DOMINIO 1: ANÁLISIS VERBAL (WHISPER + GPT-4o) ---
    print("Iniciando transcripción de audio...")
    transcription = model_stt.transcribe(path, language="es")["text"]
    
    print("Consultando a GPT-4o para evaluación IECCT-SV...")
    prompt = f"""
    Evalúa la calidez de esta teleconsulta en El Salvador usando el instrumento IECCT-SV.
    Transcripción: "{transcription}"
    
    Analiza y califica de 1 a 5:
    1. Acomodación del lenguaje.
    2. Validación emocional.
    3. Respeto y cercanía.
    
    Responde estrictamente en este formato JSON:
    {{
      "acomodacion": 0,
      "validacion": 0,
      "respeto": 0,
      "resumen": "Resumen breve del análisis verbal."
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    verbal_result = json.loads(response.choices[0].message.content)

    return {
        "verbal": verbal_result,
        "no_verbal": score_nv,
        "texto": transcription
    }

@app.route('/auditar', methods=['POST'])
def auditar():
    if 'video' not in request.files:
        return jsonify({"error": "No se recibió archivo de video"}), 400
    
    file = request.files['video']
    temp_path = "video_temp.mp4"
    file.save(temp_path)
    
    try:
        resultado = analizar_video(temp_path)
        return jsonify(resultado)
    except Exception as e:
        print(f"Error en auditoría: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Puerto dinámico para Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
