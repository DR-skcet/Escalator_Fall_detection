import cv2
import numpy as np
import matplotlib.pyplot as plt
import winsound

# Taking video feed
## EDIT THE LOCATION OF THE VIDEO
Video = cv2.VideoCapture(r'C:\Users\91934\Desktop\CCTV_Fall_Detector\docs\$RMY60OG.mp4')

# Creating a Background Substractor kernel
fgbg = cv2.createBackgroundSubtractorKNN(128, cv2.THRESH_BINARY, 1)
w_list = []
h_list = []
frame_skip = 2  # Number of frames to skip
delay = 30  # Delay in milliseconds
fall_detected = False

while True:
    # Skip frames
    for _ in range(frame_skip):
        ret = Video.grab()
        if not ret:
            break
    else:
        # Obtain frame
        ret, frame = Video.retrieve()
        if not ret:
            break

        # Convert the frame into Grey
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the grey frame
        frame_blur = cv2.GaussianBlur(frame_grey, (3, 3), 0)
        # Apply the Background Substraction mask
        fgmask = fgbg.apply(frame_blur)
        fgmask[fgmask == 127] = 0
        # Threshold the Background Substracted frame to remove grey values
        (threshold, frame_bw) = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Dilate the frame
        frame_bw = cv2.dilate(frame_bw, None, iterations=2)
        # Find the contours
        contours, hierarchy = cv2.findContours(frame_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Develop a list of large contours
        contours_thresholded = []
        # Threshold the area for determining a contour is large enough or not
        threshold_area = 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > threshold_area:
                contours_thresholded.append(cnt)

        # Sort the largest contours
        largest_contours = sorted(contours_thresholded, key=cv2.contourArea)[-5:]

        # Determine the biggest contour
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        # Determine the co-ordinates of the bounding box for the biggest contour
        (xx, yy, ww, hh) = cv2.boundingRect(biggest_contour)

        # Determine the number of largest contours
        LENGTH = len(largest_contours)

        if LENGTH > 0:
            # list of difference between the two legs for all the frames
            w_list.append(ww)
            c = cv2.rectangle(fgmask, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 1)
            # Leg 1
            cv2.circle(frame, (xx + ww, yy + hh), 2, (0, 255, 0), -1)
            # Leg 2
            cv2.circle(frame, (xx, yy + hh), 2, (255, 0, 0), -1)

            # Beep Sound when the person is falling
            if not fall_detected and len(h_list) > 0:
                last_height = h_list[-1]
                if hh < 0.7 * last_height:
                    # Beep
                    winsound.Beep(2500, 2000)
                    print("Fall Detected!")
                    fall_detected = True

            # List of height of the bounding box for all frames
            h_list.append(hh)

        # The feed showing the bounding box
        cv2.imshow('Bounding box', c)
        # The feed showing the feet points
        cv2.imshow('Security feed', frame)

        # Add a delay to slow down the video playback
        cv2.waitKey(delay)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

Video.release()
cv2.destroyAllWindows()

# Plot the difference between both feet
plt.plot(w_list)
plt.xticks(np.arange(0, 100, 2))
plt.ylim(0, 400)
plt.xlim(0, 100)
plt.show()
