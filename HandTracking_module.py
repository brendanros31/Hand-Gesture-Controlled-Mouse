import mediapipe as mp
import cv2
import time
import math
import numpy as np


class handDetector():

    def __init__(self, mode=False, maxHands=2, model_complexity = 1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence 
        self.tracking_confidence = tracking_confidence 

        # Video Capture, Hand detection - Configurations
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            self.model_complexity, 
            self.detection_confidence,
            self.tracking_confidence,
        )

        # Hand Landmark points Drawing - Initialization
        self.mpDraw = mp.solutions.drawing_utils

        # tipIds
        self.tipIds = [4, 8, 12, 16, 20]


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)   

        # hand_Landmarks - Information Extraction
        if self.results.multi_hand_landmarks:
            for hand_Landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Points, Connections - Displaying 
                    self.mpDraw.draw_landmarks(img, hand_Landmarks, self.mpHands.HAND_CONNECTIONS)
                    
        return img


    def fingersUp(self):
        fingers = []

         # Thumb
        if self.Landmark_list[self.tipIds[0]][1] > self.Landmark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Finger
        for id in range(1, 5):
            if self.Landmark_list[self.tipIds[id]][2] < self.Landmark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # total_fingers = fingers.count(1)
        
        return fingers   


    def findPosition(self, img, handNo=0, draw=True, pointNo=False):   
        xList = []
        yList = []
        bbox = []
        self.Landmark_list = []

        # Assigning Hand Numbers for tracking
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            # ID, Landmarks - Displaying 
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h,w,c = img.shape   # Height, Width, Channels of image
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)  # Append x-coordinate
                yList.append(cy)  # Append y-coordinate
                #print(id, cx, cy)
                self.Landmark_list.append([id, cx, cy])


                # Display the ID number and coordinates on the image
                if pointNo:
                    cv2.putText(img, 
                                str(id), 
                                (cx, cy),   # Position
                                cv2.FONT_HERSHEY_DUPLEX,   # Font 
                                3,   # Scale
                                (0, 255, 0),   # Color
                                2,   # Thickness
                    )
                    print(id, cx, cy)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            # BBOX - Displaying
            if draw:
                cv2.rectangle(img, 
                              (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20),
                              (0,255,0),
                              2,
                )
        
        return self.Landmark_list, bbox


    def findDistance(self, p1, p2, img, draw=True, r=5, t=2):
        # Check if Landmark_list has enough points
        if len(self.Landmark_list) <= max(p1, p2):
            #print(f"Debug: Landmark_list does not have enough points for p1={p1} or p2={p2}.")
            return 0, img, [0, 0, 0, 0, 0, 0]

        # Extract coordinates
        x1, y1 = self.Landmark_list[p1][1:]
        x2, y2 = self.Landmark_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw the line and circles if required
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        # Calculate the distance
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]



def main():
        
    # FPS - Initialization
    pTime = 0
    cTime = 0

    # Video Capture, Hand detection - Configurations
    cap = cv2.VideoCapture(0)

    # Object - Declaration
    detector = handDetector()


    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)

        if not success:
            print("Debug: Failed to capture frame from camera.")
            continue

        # Landmarks, Bbox - Hand
        Landmark_list, bbox = detector.findPosition(img)

        # FPS - Configurations
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        # FPS Text - Displaying
        cv2.putText(img, 
                    str(int(fps)), 
                    (10,70),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.5,
                    (255,255,255),
                    1,
        )

        # Result Image - Displaying
        cv2.imshow('Image', img)

        # Capture termination 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    # CLEANUP - Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

