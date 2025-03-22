from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
model = load_model('./model/model_new_50.keras')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}


def predict_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // 15, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (64, 64)) / \
                255.0
            frames.append(frame)
        if len(frames) >= 15:
            break
    cap.release()

    if len(frames) == 0:
        return False

    # ضمان نفس أبعاد التدريب
    preds = model.predict(np.array([frames]))
    return preds[0][0] > 0.5


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            file_size = round(os.path.getsize(filepath) / (1024 * 1024), 2)
            file_type = filename.rsplit('.', 1)[1].upper()
            is_violent = predict_video(filepath)

            # Pass parameters as URL query arguments
            return redirect(url_for('result',
                                    result='violent' if is_violent else 'non-violent',
                                    filename=filename,
                                    filesize=f"{file_size} MB",
                                    filetype=file_type))
        except Exception as e:
            return f"Error: {str(e)}"

    return redirect(url_for('index'))


@app.route('/result')
def result():
    return render_template('result.html',
                           result=request.args.get('result'),
                           filename=request.args.get('filename'),
                           filesize=request.args.get('filesize'),
                           filetype=request.args.get('filetype'))


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
