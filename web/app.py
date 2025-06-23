from flask import Flask, render_template, request, jsonify, Response
from audio import OLAEngine,PVEngine,HybridEngine
from userinfo import get_user_info  # Your provided code in a file called userinfo.py
import numpy as np
import os, wave
import random
import glob

engine_names = None
engines = None

app = Flask(__name__)

sessions = {}

@app.route("/")
def index():
    global engines,engine_names
    # Example session info to pass to template
    session_id = "12345"
    filename = glob.glob("samples/*.wav")[0]  # Your audio file path
    # engine_names = ["HybridEngine"]       # You can add more engines here
    
    engine_names = ["HybridEngine","OLAEngine","PVEngine"]       # You can add more engines here
    engines = [HybridEngine(filename), OLAEngine(filename), PVEngine(filename)]
    # engine_files = [f"{engine}/{filename}" for engine in engine_names]
    engine_files=[]
    for i in engine_names:
        engine_files.append(f"samples/{i}_{filename.split('/')[1]}")
        # os.makedirs(f"/tmp/{i}", exist_ok=True)
        # if not os.path.exists(f"/tmp/{i}/{filename}"):
        import shutil
        shutil.copyfile(filename, f"samples/{i}_{filename.split('/')[1]}",)

    return render_template("index.html",
                           session_id=session_id,
                           filename=filename,
                           engine_names=engine_names,
                           engine_files=engine_files,
                           zip=zip,
                           )

def generate_wav_header(sample_rate, n_channels=1, sampwidth=2, n_frames=0):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(b'')  # just header
    return buffer.getvalue()

# CHUNK_SIZE = 2048  # in bytes

# @app.route("/stream/<path:filename>")
# def stream(filename):
#     # print(os.listdir("/tmp/HybridEngine/samples"))
#     # if filename.startswith("tmp/"):
#     #     filename = filename[len("tmp/"):]  # remove leading "tmp/" from filename

#     filepath = filename
#     def generate():
#         with wave.open(filepath, 'rb') as wf:
#             header = generate_wav_header(
#                 wf.getframerate(),
#                 wf.getnchannels(),
#                 wf.getsampwidth()
#             )
#             yield header
#             data = wf.readframes(CHUNK_SIZE)
#             while data:
#                 yield data
#                 data = wf.readframes(CHUNK_SIZE)
#     return Response(generate(), mimetype="audio/wav")

@app.route("/stream/<name>/<filename>")
def stream_with_engine(name, filename):
    index = engine_names.index(name)
    engine_obj = engines[index]

    # Optional: sanitize or validate filename
    full_path = os.path.join("samples", filename.split("/")[-1])
    # engine_obj.reset(full_path)

    def generate():
        header = generate_wav_header(
            sample_rate=engine_obj.audio_sr,
            n_channels=1,
            sampwidth=2
        )
        yield header
        for chunk in engine_obj.run_generator():
            yield chunk

    return Response(generate(), mimetype="audio/wav")

# @app.route("/stream/<name>/<filename>")
# def stream_with_engine(name, filename):
#     index = engine_names.index(name)
#     engine_obj = engines[index]

#     def generate():
#         header = generate_wav_header(
#             sample_rate=engine_obj.audio_sr,
#             n_channels=1,
#             sampwidth=2
#         )
#         yield header
#         for chunk in engine_obj.run_generator():
#             yield chunk

#     return Response(generate(), mimetype="audio/wav")



@app.route("/update_speed", methods=["POST"])
def update_speed():
    data = request.get_json()
    index = data["index"]
    speed = float(data["speed"])
    filename = data["filename"]
    
    engines[index].set_alpha(speed)
    # englin
    # print("ENGINE",engine_names[index])
    
    # Return the new file path for reloading
    return jsonify({
        "message": f"{engine_names[index]} updated with speed {speed}",
        # "new_url": f"/stream{output_file}"
    })


if __name__ == "__main__":
    app.run(debug=True, threaded=True)