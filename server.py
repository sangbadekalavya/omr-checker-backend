from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

ANSWER_OPTIONS = ["A", "B", "C", "D"]

@app.route('/read-omr', methods=['POST'])
def read_omr():

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    npimg = np.frombuffer(file.read(), np.uint8)

    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bubble_contours = []

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)

        aspect_ratio = w / float(h)

        if w >= 20 and h >= 20 and 0.8 <= aspect_ratio <= 1.2:
            bubble_contours.append(c)

    bubble_contours = sorted(
        bubble_contours,
        key=lambda c: cv2.boundingRect(c)[1]
    )

    answers = []

    question_count = len(bubble_contours) // 4

    for q in range(question_count):

        row = bubble_contours[q * 4:(q + 1) * 4]

        row = sorted(
            row,
            key=lambda c: cv2.boundingRect(c)[0]
        )

        bubbled = None
        max_pixels = 0

        for j, c in enumerate(row):

            mask = np.zeros(thresh.shape, dtype="uint8")

            cv2.drawContours(mask, [c], -1, 255, -1)

            masked = cv2.bitwise_and(
                thresh,
                thresh,
                mask=mask
            )

            total = cv2.countNonZero(masked)

            if total > max_pixels:
                max_pixels = total
                bubbled = j

        if bubbled is not None:
            answers.append(ANSWER_OPTIONS[bubbled])
        else:
            answers.append("-")

    return jsonify({
        "answers": answers
    })

@app.route('/')
def home():
    return "OMR Backend Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
