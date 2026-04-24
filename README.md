# Face Recognition Project

This is a Python-based project that uses the `face_recognition` and `opencv-python` libraries to detect and identify faces both in static images and through a live webcam feed. 

## Features
- **Live Camera Recognition** (`camera_test.py`): Real-time face detection and matching using your computer's webcam.
- **Static Image Recognition** (`face_test.py`): Detect and identify faces within an image file.
- **Dynamic Database**: Automatically creates a `database` directory to store known faces. Simply drop images into this folder (named after the person, e.g., `john_doe.jpg`), and the script will automatically learn and recognize them.

## Prerequisites & Installation

### For Mac and Linux:
Installation is straightforward. Run:
```bash
pip install -r requirements.txt
```

### For Windows Users (Important ⚠️):
The `face_recognition` library relies on `dlib`, which requires C++ build tools to compile on Windows.
1. Download and install **Visual Studio Build Tools** from the Microsoft website.
2. During installation, select **Desktop development with C++**.
3. Once installed, restart your computer.
4. Open your terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Add Known Faces**: Place an image of the person you want to recognize in the `database` folder. Name the file with the person's name (e.g., `robert_downey_jr.jpg`).
2. **Run Image Test**: 
   Ensure you have an image named `test1.jpg` in the root directory, then run:
   ```bash
   python face_test.py
   ```
3. **Run Live Webcam Test**:
   ```bash
   python camera_test.py
   ```
   Press `q` to exit the live webcam feed.
