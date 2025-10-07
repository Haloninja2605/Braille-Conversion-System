import cv2
import numpy as np
from picamera2 import Picamera2
from time import sleep
from quaddetection import detect_quad
from keras.models import load_model
import imutils
from imutils.contours import sort_contours
import RPi.GPIO as GPIO

# ----- Hardware Setup -----
# Servo Configuration
SERVO_PIN = 18
# Braille Pins (Two 3x2 Displays: 12 Total Pins)
BRAILLE_DISPLAY_1 = [2, 3, 4, 5, 6, 7]  # Pins for Display 1
BRAILLE_DISPLAY_2 = [8, 9, 10, 11, 12, 13]    # Pins for Display 2

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
for pin in BRAILLE_DISPLAY_1 + BRAILLE_DISPLAY_2:
    GPIO.setup(pin, GPIO.OUT)

# Servo PWM Setup
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

# Braille Mapping (0-9)
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

# ----- Helper Functions -----
def set_servo_angle(angle):
    """Move servo to a safe angle (0-90°)."""
    angle = max(0, min(90, angle))
    duty = (angle / 90) * (12.5 - 2.5) + 2.5  # Map 0-90° to 2.5-12.5%
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

def activate_braille(group):
    """Activate Braille displays for a group (1 or 2 digits)."""
    # Reset all pins
    for pin in BRAILLE_DISPLAY_1 + BRAILLE_DISPLAY_2:
        GPIO.output(pin, GPIO.LOW)
    
    # Activate pins for each digit in the group
    for i, digit in enumerate(group[:2]):  # Process max 2 digits
        braille = braille_digit_map.get(str(digit), [[0,0],[0,0],[0,0]])
        states = [braille[row][col] for row in range(3) for col in range(2)]
        target_pins = BRAILLE_DISPLAY_1 if i == 0 else BRAILLE_DISPLAY_2
        for pin, state in zip(target_pins, states):
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def quadtree_segmentation(path, threshold=15, min_size=15):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    class Node:
        def __init__(self, x0, y0, w, h):
            self.x0, self.y0, self.w, self.h = x0, y0, w, h
            self.children = []
        def region(self): return img[self.y0:self.y0+self.h, self.x0:self.x0+self.w]
        def error(self): return np.std(self.region())
    def split(node, nodes):
        if node.error() > threshold and node.w > min_size and node.h > min_size:
            w2, h2 = node.w//2, node.h//2
            for dx, dy in [(0,0),(w2,0),(0,h2),(w2,h2)]:
                child = Node(node.x0+dx, node.y0+dy, w2, h2)
                node.children.append(child)
                split(child, nodes)
        else:
            nodes.append(node)
    root = Node(0,0,w,h)
    leaves = []
    split(root, leaves)
    background = np.zeros_like(img)
    for node in leaves:
        region = node.region()
        background[node.y0:node.y0+node.h, node.x0:node.x0+node.w] = np.mean(region)
    foreground = cv2.absdiff(img, background.astype(np.uint8))
    _, foreground = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    foreground = cv2.medianBlur(foreground, 5)
    kernel = np.ones((3,3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Background', background.astype(np.uint8))
    cv2.imshow('Foreground', foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return background, foreground

def non_max_suppression(boxes, overlap_thresh=0.3):
    """Filter overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]
    area = (boxes[:,2]) * (boxes[:,3])
    idxs = np.argsort(area)
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")

def cluster_and_sort_contours(cnts, row_tolerance=20):
    """Sort contours top-to-bottom and left-to-right within rows."""
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = non_max_suppression(boxes)  # Remove overlaps
    
    # Cluster into rows
    y_coords = np.array([y for (x, y, w, h) in boxes])
    sorted_indices = np.argsort(y_coords)
    current_y = y_coords[sorted_indices[0]]
    rows = []
    row = []
    
    for i in sorted_indices:
        x, y, w, h = boxes[i]
        if abs(y - current_y) > row_tolerance:
            rows.append(sorted(row, key=lambda b: b[0]))  # Sort row left-to-right
            row = [(x, y, w, h)]
            current_y = y
        else:
            row.append((x, y, w, h))
    if row:
        rows.append(sorted(row, key=lambda b: b[0]))
    
    # Flatten rows into final order
    sorted_boxes = [box for row in rows for box in row]
    return sorted_boxes

# ----- Model & Main Logic -----
model = load_model("model.h5")


def main():
    # Capture Image
    picam2 = Picamera2()
    picam2.start()
    sleep(2)
    frame = picam2.capture_array()
    picam2.stop()
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite('capture.jpg', image)

    # Detect Quad and Preprocess
    warped_thresh = detect_quad(image, 640, 480)
    if warped_thresh is None:
        print("Quad not detected.")
        return
    cv2.imwrite('scanned.jpg', warped_thresh)
    
    # Quadtree decomposition
    bg, fg = quadtree_segmentation('scanned.jpg')
    gray = cv2.bitwise_not(warped_thresh)  # Invert once here
    image = cv2.cvtColor(warped_thresh, cv2.COLOR_GRAY2BGR)  # Convert to color for annotations

    # Digit Extraction
    edged = cv2.Canny(gray, 30, 150)
    cnts = imutils.grab_contours(cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
    
    # Sort contours
    boxes = cluster_and_sort_contours(cnts) if cnts else []
    
    chars, digits = [], []
    for (x, y, w, h) in boxes:
        if w < 8 or h < 15 or w > 140 or h > 140:
            continue
        roi = gray[y:y+h, x:x+w]
        
        # --- FULL PREPROCESSING ---
        thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        (H, W) = thr.shape
        if W > H:
            thr = imutils.resize(thr, width=20)
            dH = int((20 - thr.shape[0]) / 2.0)
            pad = cv2.copyMakeBorder(thr, top=dH, bottom=dH, left=4, right=4, 
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        else:
            thr = imutils.resize(thr, height=20)
            dW = int((20 - thr.shape[1]) / 2.0)
            pad = cv2.copyMakeBorder(thr, top=4, bottom=4, left=dW, right=dW,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        pad = cv2.resize(pad, (28, 28))
        pad = pad.astype('float32') / 255.0
        chars.append(np.expand_dims(pad, axis=-1))

    # Step 3: Predict and group digits
    if chars:
        preds = model.predict(np.array(chars))
        digits = [np.argmax(pred) for pred in preds]
        # Draw boxes in sorted order
        for (digit, (x, y, w, h)) in zip(digits, boxes):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    print(digits)
    # Step 4: Display groups sequentially
    if digits:
        groups = [digits[i:i+2] for i in range(0, len(digits), 2)]
        for group in groups:
            activate_braille(group)
            if group:
                set_servo_angle(group[0] * 10)
            cv2.imshow('Result', image)
            cv2.waitKey(3000) # Show each group for 3 seconds

    # Cleanup
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    servo.stop()
    GPIO.cleanup()

if __name__ == '__main__':
    main()