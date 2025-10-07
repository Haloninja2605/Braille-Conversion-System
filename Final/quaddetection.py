import cv2
import numpy as np
from utlis import initializeTrackbars, valTrackbars, biggestContour, reorder, stackImages


def detect_quad(image, heightImg=640, widthImg=480):
    """
    Uses a static image and OpenCV trackbars to adjust Canny thresholds live.
    Press 's' to capture and return the warped gray image, or 'q' to cancel.

    Args:
        image (np.ndarray): BGR input image.
        heightImg (int): height to resize for processing.
        widthImg (int): width to resize for processing.

    Returns:
        warped_gray (np.ndarray or None): warped grayscale scan if 's' pressed, None if 'q'.
    """
    # 1. Initialize trackbars
    initializeTrackbars()

    warped_gray = None
    while True:
        # 2. Resize and preprocess
        img = cv2.resize(image, (widthImg, heightImg))
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)

        # 3. Get threshold values
        t1, t2 = valTrackbars()
        imgCanny = cv2.Canny(imgBlur, t1, t2)
        kernel = np.ones((5,5))
        imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
        imgThresh = cv2.erode(imgDial, kernel, iterations=1)

        # 4. Draw all contours
        contours, _ = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imgContours = img.copy()
        cv2.drawContours(imgContours, contours, -1, (0,255,0), 2)

        # 5. Extract and warp biggest contour
        biggest, _ = biggestContour(contours)
        imgBig = img.copy()
        warp = img.copy()
        warped_gray = np.zeros((heightImg, widthImg), np.uint8)

        if biggest.size:
            biggest = reorder(biggest)
            cv2.drawContours(imgBig, biggest, -1, (0,255,0), 5)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            warp = cv2.warpPerspective(img, M, (widthImg, heightImg))
            warp = warp[20:-20, 20:-20]
            warp = cv2.resize(warp, (widthImg, heightImg))
            warped_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

        # 6. Overlay threshold values and display stacked views
        # show numeric thresholds on the contours image
        thresh_overlay = imgContours.copy()
        cv2.putText(thresh_overlay, f"T1:{t1} T2:{t2}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # also show the individual thresholded image in its own window for clarity
        cv2.imshow("Thresholded", imgThresh)

        arr = ([img, imgGray, imgThresh, thresh_overlay],
               [imgBig, warp, warped_gray, warped_gray])
        labels = [["Orig","Gray","Thresh","Cntrs"],
                  ["Biggest","Warp","WarpGray","WarpGray"]]
        vis = stackImages(arr, 0.7, labels)
        cv2.imshow("Quad Detection", vis)
        arr = ([img, imgGray, imgThresh, imgContours],
               [imgBig, warp, warped_gray, warped_gray])
        labels = [["Orig","Gray","Thresh","Cntrs"],
                  ["Biggest","Warp","WarpGray","WarpGray"]]
        vis = stackImages(arr, 0.7, labels)
        cv2.imshow("Quad Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            warped_gray = None
            break

    cv2.destroyAllWindows()
    return warped_gray


if __name__ == "__main__":
    # Demo mode: load static image
    pathImage = "test1.jpeg"
    image = cv2.imread(pathImage)
    if image is None:
        print(f"ERROR: cannot load '{pathImage}'")
        exit(1)
    result = detect_quad(image)
    if result is not None:
        cv2.imwrite("scanned_demo.jpg", result)
        print("Saved scanned_demo.jpg")
    else:
        print("Detection cancelled.")
