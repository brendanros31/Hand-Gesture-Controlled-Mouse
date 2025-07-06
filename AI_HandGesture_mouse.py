import cv2
import numpy as np
import HandTracking_module as htm
import time
import autopy


# Camera - Screen Size Declaration
cam_width, cam_height = 640, 480
screen_width, screen_height = autopy.screen.size()

# Screen - Dimentions
frameRedn = 100   # Frame displayed on screen
buffer = 60   # Pixels to exclude from the edges of the camera feed

# Capture - Initialization
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

# FPS - Initialization
pTime = 0

# Hand Detector - Declaration
detector = htm.handDetector(maxHands=1)

# Object - Initialization
fingers = []
smoothening = 5  # Smoothening values

# Pointer locations
xPrev, yPrev = 0, 0
xCurr, yCurr = 0, 0



while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if not success:
        print("Debug: Failed to capture frame from camera.")
        time.sleep(1)  # Delay before retrying capture
        continue


# Landmarks
    # Hand
    img = detector.findHands(img)
    Landmark_list, bbox = detector.findPosition(img)    

    # Tip of Index and Thumb
    if len(Landmark_list) != 0:
        x1, y1 = Landmark_list[8][1:]   # Index Finger
        x2, y2 = Landmark_list[4][1:]   # Thumb
        #print(x1, y1, x2, y2)


# Checking which finger is up
        fingers = detector.fingersUp()
        #print(fingers)


# Pointer - Screen Interaction
    if len(fingers) > 1 and fingers[1] == 1 and fingers[2] == 0:

        # Frame for pointer movement
        cv2.rectangle(img, 
                    (frameRedn, frameRedn), 
                    (cam_width-frameRedn, cam_height-frameRedn),
                    (255,255,255),
                    2,
        )

        # Final screen Corelation - Sensitivity
        Redn = frameRedn+buffer 

        # Converting Coordinates for Pointer movement
        x3 = np.interp(x1, (Redn, cam_width-Redn), (0, screen_width))
        y3 = np.interp(y1, (Redn, cam_height-Redn), (0, screen_height))

        # Clamp coordinates to ensure they are within screen bounds
        x3 = max(0, min(screen_width - 1, x3))
        y3 = max(0, min(screen_height - 1, y3))

        # Smoothening values
        xCurr = xPrev + (x3-xPrev)/smoothening
        yCurr = yPrev + (y3-yPrev)/smoothening

        # REGISTERING - Pointer movement
        autopy.mouse.move(xCurr, yCurr)
        cv2.circle(img, (x1,y1), 5, (255,255,255), cv2.FILLED)
        xPrev, yPrev = xCurr, yCurr


# 'Click' Interaction
    if len(fingers) > 1 and fingers[1] == 1 and fingers[2] == 0:
        length, img, lineInfo = detector.findDistance(4, 8, img)
        #print(length)
    
        # Checking distance between fingers
        if length < 20:   # 20 is the length chosen between fingers to consider a Click
            cv2.circle(img, (lineInfo[4],lineInfo[5]), 5, (0,250,0), cv2.FILLED)

            # REGISTERING - Click
            autopy.mouse.click()


# On Screen Text 
    # FPS - Configurations
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # FPS Text - Displaying
    cv2.putText(img, 
                f'FPS: {int(fps)}', 
                (20, 50),   # Position
                cv2.FONT_HERSHEY_DUPLEX,   # Font
                0.5,   # Scale
                (255, 255, 255),   # Color
                1,   # Thickness
    )


# Product 
    # Result Image - Display
    cv2.imshow('Image', img)

    # Capture termination 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break



# CLEANUP - Release resources and close windows
cap.release()
cv2.destroyAllWindows()