from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from keras.api.models import load_model
from keras.api.applications.densenet import preprocess_input
from skimage.io import imread

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configuração para servir o diretório de uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Configuração para servir o diretório estático
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Carregue seu modelo aqui
model = load_model('modelo_raca_caninas.h5')

breed_list = os.listdir("Images/")

# Mapeamento de rótulos
label_maps_rev = {i: v for i, v in enumerate(breed_list)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = previsao_foto(filepath)
        return render_template('index.html', result=result)
    return redirect(request.url)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpeg', 'jpg'}

def previsao_foto(filepath):
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img.save(filepath)
    img = imread(filepath)
    img = preprocess_input(img)
    probs = model.predict(np.expand_dims(img, axis=0))
    predictions = []
    for idx in np.argsort(probs[0])[::-1][:5]:
        try:
            label = label_maps_rev[idx]
            predictions.append((f"{probs[0][idx]*100:.2f}%", label))
        except KeyError:
            predictions.append((f"{probs[0][idx]*100:.2f}%", f"Label {idx} não mapeado"))
    image_url = f"/uploads/{os.path.basename(filepath)}"
    return {'image_url': image_url, 'predictions': predictions}

if __name__ == '__main__':
    app.run(debug=True)
