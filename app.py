from flask import Flask, request, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# YOLOv4 모델 로딩
yolov4_weights = 'yolov4.weights'
yolov4_cfg = 'yolov4.cfg'
net = cv2.dnn.readNet(yolov4_weights, yolov4_cfg)

# 클래스 이름 설정
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return "No image provided"

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    height, width, _ = frame.shape
    conf_threshold = 0.5
    nms_threshold = 0.4

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in range(len(boxes)):
        for i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    _, img_encoded = cv2.imencode('.jpg', frame)
    response = img_encoded.tobytes()

    return Response(response, content_type='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
