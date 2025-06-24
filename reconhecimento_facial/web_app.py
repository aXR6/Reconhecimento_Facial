from flask import Flask, request, jsonify

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"

from reconhecimento_facial.face_detection import detect_faces
from reconhecimento_facial.llm_service import generate_caption
from reconhecimento_facial.obstruction_detection import detect_obstruction
from reconhecimento_facial.db import list_people, list_detections

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
    return jsonify({'faces': faces, 'caption': caption, 'obstruction': obstruction})


@app.route('/people', methods=['GET'])
def people():
    """List registered people."""
    return jsonify({'people': list_people()})


@app.route('/detections', methods=['GET'])
def detections():
    """List recent detections."""
    rows = list_detections(100)
    res = [
        {
            'id': r[0],
            'image': r[1],
            'faces': r[2],
            'caption': r[3],
            'obstruction': r[4],
            'recognized': r[5],
            'created_at': r[6].isoformat() if hasattr(r[6], "isoformat") else r[6],
        }
        for r in rows
    ]
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
