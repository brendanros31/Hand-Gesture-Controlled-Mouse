# AI Hand Gesture Mouse Controller

This is a computer vision-based project that allows users to control their computer's mouse using hand gestures captured via a webcam. 

Built using [MediaPipe](https://mediapipe.dev/) and OpenCV, the system translates hand gestures into real-time mouse movements and actions, providing a touch-free interaction experience.


## Features

- **Real-Time Hand Tracking** using MediaPipe.
- **Cursor Movement** based on the index finger's tip.
- **Click Gesture** using index and thumb finger proximity.
- **Click-and-Drag (Pinch) Gesture** using thumb and index finger.
- **Edge Bounding** to prevent cursor flickering near screen edges.
- Modularized with clean, reusable functions in `HandTracking_module.py`.


## How to Use
```bash
pip install opencv-python mediapipe pyautogui numpy
git clone https://github.com/brendanros31/Hand-Gesture-Controlled-Mouse.git
cd Hand-Gesture-Controlled-Mouse
python AI_HandGesture_mouse.py

```


## Project Structure
```
./
├── AI_HandGesture_mouse.py       # Main script to run the gesture-based mouse
├── HandTracking_module.py        # Custom module for hand tracking logic
├── README.md                     # Project documentation (this file)