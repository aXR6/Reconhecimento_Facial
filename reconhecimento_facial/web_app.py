from flask import Flask, request, jsonify

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"

from reconhecimento_facial.face_detection import detect_faces
from reconhecimento_facial.llm_service import generate_caption
from reconhecimento_facial.obstruction_detection import detect_obstruction
from reconhecimento_facial.emotion_detection import detect_emotion

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process():
    file = request.files.get('file')
    if not file:
        return {'error': 'no file'}, 400
    path = '/tmp/upload.jpg'
    file.save(path)
    faces = detect_faces(path, path, as_json=True, save_db=True)
    caption = generate_caption(path)
    obstruction = detect_obstruction(path)
    emotion = detect_emotion(path)
    return jsonify({'faces': faces, 'caption': caption, 'obstruction': obstruction, 'emotion': emotion})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
