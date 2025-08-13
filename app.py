import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class handDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []
        fingers.append(1 if self.lmList[4][1] > self.lmList[3][1] else 0)
        for id in range(1, 5):
            fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] else 0)
        return fingers


class AirDrawTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = handDetector()
        self.canvas = None
        self.xp, self.yp = 0, 0
        self.brushColor = (255, 0, 255)

        # Color buttons: (coords, color, name)
        self.colorButtons = [
            ((10, 10, 110, 60), (255, 0, 255), "Pink"),
            ((120, 10, 220, 60), (255, 0, 0), "Blue"),
            ((230, 10, 330, 60), (0, 255, 0), "Green"),
            ((340, 10, 440, 60), (0, 255, 255), "Yellow"),
            ((450, 10, 550, 60), (0, 0, 0), "Eraser")
        ]

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # Draw color buttons
        for (x1, y1, x2, y2), color, name in self.colorButtons:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            fingers = self.detector.fingersUp()
            x1, y1 = lmList[8][1:]  # Index finger tip

            if fingers[1] == 1 and fingers[2] == 0:
                # Color selection
                for (x1b, y1b, x2b, y2b), color, _ in self.colorButtons:
                    if x1b < x1 < x2b and y1b < y1 < y2b:
                        self.brushColor = color
                        self.xp, self.yp = 0, 0
                        break
                else:
                    # Drawing mode
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1
                    thickness = 30 if self.brushColor == (0, 0, 0) else 8
                    cv2.line(self.canvas, (self.xp, self.yp), (x1, y1),
                             self.brushColor, thickness)
                    self.xp, self.yp = x1, y1
            else:
                self.xp, self.yp = 0, 0

        # Merge drawings
        imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, self.canvas)

        return img


st.title("✏️ Air Drawing with Colors - Streamlit Edition")
st.markdown("Draw in the air using your hand, choose colors from the palette above.")

webrtc_streamer(key="airdraw",
                video_transformer_factory=AirDrawTransformer,
                media_stream_constraints={"video": True, "audio": False})
