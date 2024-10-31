import cv2
import numpy as np

# print("Himanshu Singh")
# print("open cv version " + cv2.__version__)  # Corrected version print

# web camera or video capture
cap = cv2.VideoCapture("C:/Users/hp/Downloads/CV_video.mp4")


# count line position
count_line_position = 550

# minimum width and height for the rectangle
min_width_rect = 80  
min_height_rect = 80

# Defining the function to get center coordinates
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []

offset = 6  # Acceptable error in pixels
counter = 0

# Initialize background subtractor
algo = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame1 = cap.read()

    if not ret:
        print("Error: Unable to read video frame")
        break

    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Applying the background subtractor on the blurred frame
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 122, 0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        val_counter = (w >= min_width_rect) and (h >= min_height_rect)

        if not val_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "VEHICLE " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (cx, cy) in detect:
            if (count_line_position - offset) < cy < (count_line_position + offset):
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((cx, cy))
                print("Vehicle Counter: " + str(counter))

    cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show video feed
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:  # Press 'Enter' to stop
        break

cv2.destroyAllWindows()  # Corrected function name
cap.release()
