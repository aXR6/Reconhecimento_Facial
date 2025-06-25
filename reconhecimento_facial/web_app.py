from flask import Flask, request, jsonify, render_template

if __package__ is None or __package__ == "":
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "reconhecimento_facial"

# Heavy modules are imported lazily inside the helper wrappers below to avoid
# importing optional dependencies when this module is imported during tests.

def detect_faces(*args, **kwargs):  # noqa: D401 - wrapper for lazy import
    """Call :func:`face_detection.detect_faces` lazily."""
    from reconhecimento_facial.face_detection import detect_faces as _df

    return _df(*args, **kwargs)


def generate_caption(*args, **kwargs):  # noqa: D401 - wrapper for lazy import
    """Call :func:`llm_service.generate_caption` lazily."""
    from reconhecimento_facial.llm_service import generate_caption as _gc

    return _gc(*args, **kwargs)


def detect_obstruction(*args, **kwargs):  # noqa: D401 - wrapper for lazy import
    """Call :func:`obstruction_detection.detect_obstruction` lazily."""
    from reconhecimento_facial.obstruction_detection import (
        detect_obstruction as _do,
    )

    return _do(*args, **kwargs)


def list_people():  # noqa: D401 - wrapper for lazy import
    """Call :func:`db.list_people` lazily."""
    from reconhecimento_facial.db import list_people as _lp

    return _lp()


def list_detections(limit: int = 100):  # noqa: D401 - wrapper for lazy import
    """Call :func:`db.list_detections` lazily."""
    from reconhecimento_facial.db import list_detections as _ld

    return _ld(limit)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


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


@app.route('/')
def index() -> str:
    """Display the home page with links to all actions."""
    return render_template('index.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect_page() -> str:
    """Upload an image and run face detection."""
    if request.method == 'POST':
        file = request.files.get('image')
        model = request.form.get('model', 'opencv')
        if not file:
            return render_template('detect.html', error='Selecione uma imagem')
        img_path = '/tmp/input.jpg'
        out_path = '/tmp/output.jpg'
        file.save(img_path)
        use_hf = model in ('mediapipe', 'yolov8')
        faces = detect_faces(img_path, out_path, use_hf=use_hf, hf_model=model)
        with open(out_path, 'rb') as fh:
            import base64

            encoded = base64.b64encode(fh.read()).decode('utf-8')
        return render_template(
            'detect.html',
            faces=faces,
            model=model,
            image=encoded,
        )
    return render_template('detect.html')


@app.route('/caption', methods=['GET', 'POST'])
def caption_page() -> str:
    """Generate an image caption using the LLM service."""
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('caption.html', error='Selecione uma imagem')
        img_path = '/tmp/caption.jpg'
        file.save(img_path)
        caption = generate_caption(img_path)
        return render_template('caption.html', caption=caption)
    return render_template('caption.html')


@app.route('/obstruction', methods=['GET', 'POST'])
def obstruction_page() -> str:
    """Detect face obstruction in an image."""
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template(
                'obstruction.html', error='Selecione uma imagem'
            )
        img_path = '/tmp/obstruction.jpg'
        file.save(img_path)
        label = detect_obstruction(img_path)
        return render_template('obstruction.html', label=label)
    return render_template('obstruction.html')


@app.route('/recognize', methods=['GET', 'POST'])
def recognize_page() -> str:
    """Recognize faces in an uploaded image."""
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('recognize.html', error='Selecione uma imagem')
        img_path = '/tmp/recognize.jpg'
        file.save(img_path)
        from reconhecimento_facial.recognition import recognize_faces

        names = recognize_faces(img_path)
        return render_template('recognize.html', names=names)
    return render_template('recognize.html')


@app.route('/register', methods=['GET', 'POST'])
def register_page() -> str:
    """Register a person by uploading a photo."""
    if request.method == 'POST':
        file = request.files.get('image')
        name = request.form.get('name', '').strip()
        if not file or not name:
            return render_template('register.html', error='Informe nome e imagem')
        img_path = '/tmp/register.jpg'
        file.save(img_path)
        from reconhecimento_facial.recognition import register_person_cli

        ok = register_person_cli(img_path, name)
        return render_template('register.html', success=ok)
    return render_template('register.html')


@app.route('/people_view')
def people_view() -> str:
    """Show people registered in the database."""
    names = list_people()
    return render_template('people.html', names=names)


@app.route('/detections_view')
def detections_view() -> str:
    """Display recent detections stored in the database."""
    rows = list_detections(20)
    return render_template('detections.html', detections=rows)


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
