import cv2
import imutils
from imutils.contours import sort_contours
from keras.models import load_model
import numpy as np
from picamera2 import Picamera2
from time import sleep

# Load the pre-trained MNIST digit model
model = load_model("model.h5")

# Label map for digits 0-9
label_map = {i: str(i) for i in range(10)}

# Braille mapping for digits 0–9 in 3x2 binary arrays
braille_digit_map = {
    '0': [[0, 1], [1, 1], [0, 0]],
    '1': [[1, 0], [0, 0], [0, 0]],
    '2': [[1, 0], [1, 0], [0, 0]],
    '3': [[1, 1], [0, 0], [0, 0]],
    '4': [[1, 1], [0, 1], [0, 0]],
    '5': [[1, 0], [0, 1], [0, 0]],
    '6': [[1, 1], [1, 0], [0, 0]],
    '7': [[1, 1], [1, 1], [0, 0]],
    '8': [[1, 0], [1, 1], [0, 0]],
    '9': [[0, 1], [1, 0], [0, 0]]
}

def process_predictions(predictions, boxes, image):
    for (pred, (x, y, w, h)) in zip(predictions, boxes):
        i = np.argmax(pred)
        if i not in label_map:
            print("[ERROR] Prediction index out of range")
            continue
        label = label_map[i]
        prob = pred[i]
        braille_array = braille_digit_map.get(label)
        print(f"[INFO] Detected: {label} - {prob * 100:.2f}%")
        print("[BRAILLE] 3x2 binary array:")
        for row in braille_array:
            print(row)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Camera Preview", image)
        cv2.waitKey(500)
        sleep(1.5)

def main():
    picam2 = Picamera2()
    picam2.start()
    try:
        while True:
            frame = picam2.capture_array()
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)

            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="top-to-bottom")[0]

            chars = []
            W = 28
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if 8 <= w <= 140 and 15 <= h <= 140:
                    roi = gray[y:y + h, x:x + w]
                    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    if thresh.shape[1] > thresh.shape[0]:
                        thresh = imutils.resize(thresh, width=W)
                    else:
                        thresh = imutils.resize(thresh, height=W)
                    (tH, tW) = thresh.shape
                    dX = int(max(0, W - tW) / 2.0)
                    dY = int(max(0, W - tH) / 2.0)
                    padded = cv2.copyMakeBorder(thresh, top=dY*2+3, bottom=dY*2+3, left=dX*2+3, right=dX*2+3,
                                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    padded = cv2.resize(padded, (W, W))
                    padded = padded.astype("float32") / 255.0
                    padded = np.expand_dims(padded, axis=-1)
                    chars.append((padded, (x, y, w, h)))

            if chars:
                boxes = [b[1] for b in chars]
                chars = np.array([c[0] for c in chars], dtype="float32")
                preds = model.predict(chars)
                process_predictions(preds, boxes, image)

            cv2.imshow("Camera Preview", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
