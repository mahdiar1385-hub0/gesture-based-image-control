# Gesture-Based Image Control

A real-time computer vision application that uses hand gestures to control image effects through a webcam.

## Features

- Track hand landmarks with MediaPipe Hands.
- Adjust brightness, saturation, and hue using the right hand.
- Control zoom and background blur using the left hand.
- Apply a portrait-style background blur with MediaPipe Selfie Segmentation.
- Display the processed webcam feed in real time.

## Requirements

- Python 3.9–3.11 recommended
- A working webcam
- The Python packages listed in `requirements.txt`

> MediaPipe compatibility can vary by Python version and operating system. If installation fails, check the MediaPipe installation documentation and use a supported Python version.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/gesture-based-image-control.git
cd gesture-based-image-control
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Allow the application to access your webcam when prompted. Press **Q** to quit.

## Controls

The application uses hand gestures to adjust image effects. Keep your hands visible to the webcam and move your thumb and index finger to change the active control.

| Hand | Control |
|---|---|
| Right | Brightness, saturation, hue |
| Left | Zoom, portrait background blur |

The exact gesture-to-effect mapping is implemented in `main.py`.

## Project Structure

```text
gesture-based-image-control/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## Notes

- Performance depends on webcam quality and hardware.
- Good lighting and a clear view of your hands improve tracking.
- This project is intended as a computer vision and human-computer interaction demo.
