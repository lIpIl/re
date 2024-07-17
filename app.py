import os
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    image = Image.open(file.stream)  # 이미지 로드
    results = model(image)  # 객체 탐지
    results.render()  # 결과 이미지에 바운딩 박스와 라벨 추가
    img_byte_arr = io.BytesIO()
    results.imgs[0].save(img_byte_arr, format='JPEG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    return jsonify({'image': encoded_img})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
