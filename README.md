# Air-Ink — Tracking MVP

This first milestone only performs webcam hand tracking and fingertip trajectory visualization.
It does **not** record samples to disk and does **not** recognize digits yet.

## Reference environment

- Python 3.11
- MediaPipe Tasks
- OpenCV

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download the pretrained hand model

```bash
python download_model.py
```

The script downloads the official MediaPipe Hand Landmarker `.task` model into `models/`.

## 3. Run the MVP

```bash
python tracking_mvp.py
```

- Show one hand to the webcam.
- The fingertip of the index finger is highlighted in red.
- The recent trajectory is drawn in green.
- Press `c` to clear the trail.
- Press `q` to quit.

If the webcam does not open, change `CAMERA_INDEX` in `tracking_mvp.py` from `0` to `1`.
Also check operating-system camera permissions and close other applications using the camera.

## Current design decisions

- One hand is tracked.
- MediaPipe landmark 8 is used as the index fingertip.
- The preview is mirrored for natural interaction.
- The trajectory is held in memory only.
- No raw video or trajectory dataset is saved in this milestone.
- A missing-hand frame inserts a trajectory break to avoid drawing a false line across tracking loss.

## Important

A visible trail is not yet a validated writing sample. Recording, start/stop controls,
trajectory metadata, Touch collection, and recognition will be separate milestones.
