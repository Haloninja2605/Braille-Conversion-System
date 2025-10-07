import cv2
import imutils
from imutils.contours import sort_contours
from keras.models import load_model
import numpy as np
from picamera2 import Picamera2
from time import sleep

# Load the pre-trained model
model = load_model("emnist_model.keras")

# EMNIST character labels
label_map = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
             10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
             20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
             30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'d', 39:'e',
             40:'f', 41:'g', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}

def process_predictions(predictions, boxes, image):
    for (pred, (x, y, w, h)) in zip(predictions, boxes):
        i = np.argmax(pred)
        if i not in label_map:
            print("[ERROR] Prediction index out of range")
            continue
        label = label_map[i]
        prob = pred[i]
        print(f"[INFO] Detected: {label} - {prob * 100:.2f}%")
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Camera Preview", image)
        cv2.waitKey(500)  # Show bounding box briefly before proceeding
        sleep(1.5)  # Increased sleep time between detections

def main():
    picam2 = Picamera2()
    picam2.start()
    try:
        while True:
            frame = picam2.capture_array()
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)  # Invert image to match black text on white background
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Increased blur to reduce noise
            edged = cv2.Canny(blurred, 30, 150)  # Adjusted Canny edge detection
            
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="top-to-bottom")[0]  # Change sorting order
            
            chars = []
            W = 28
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if 8 <= w <= 140 and 15 <= h <= 140:  # Reduced strictness for bounding boxes
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
