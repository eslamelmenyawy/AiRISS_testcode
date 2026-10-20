<div align="center">
  <h1>AiRISS — Image Inspection Toolkit</h1>

  <p>A lightweight desktop tool for inspecting industrial images: open 8–16-bit images, select regions of interest, crop, normalise and analyse histograms.</p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
    <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.9-green">
    <img alt="Tkinter" src="https://img.shields.io/badge/GUI-Tkinter-orange">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
  </p>
</div>

---

## Overview

Machine-vision cameras often produce **10, 12 or 16-bit images** that ordinary image viewers can't display correctly. AiRISS loads them unchanged, converts them for display, and gives quick tools for the first step of building an inspection system: looking closely at the region where defects appear and checking its intensity distribution.

## Features

| Menu | Feature | Purpose |
|---|---|---|
| File | **Open / Save** | Load 8–16-bit images unchanged (`cv2.IMREAD_UNCHANGED`) and save results |
| Algorithm | **Select ROI** | Drag a box with the mouse to mark a region of interest |
| Algorithm | **Crop** | Extract the selected ROI |
| Algorithm | **Normalize** | Stretch the ROI's contrast to the full intensity range |
| Algorithm | **Histogram** | Plot the intensity histogram of the selected region |
| Actions | **Zoom In / Out** | Inspect fine detail |
| — | **Live footer** | Shows cursor position and pixel information |

## Quick start

```bash
git clone https://github.com/eslamelmenyawy/AiRISS_testcode.git
cd AiRISS_testcode

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r src/requirements.txt
python src/main.py
```

## Typical workflow

1. **File → Open** an image from your inspection camera.
2. **Algorithm → Select ROI** and drag a box around the area of interest.
3. **Crop**, then **Normalize** to make low-contrast defects visible.
4. **Histogram** to check the intensity distribution, which helps choose thresholds for classical CV or decide on exposure settings.
5. **File → Save** the processed region.

## Project structure

```
AiRISS_testcode/
├── src/
│   ├── main.py            # Tkinter application (ImageProcessorApp)
│   └── requirements.txt
├── LICENSE
└── README.md
```

## License

Released under the [MIT License](LICENSE).

---

<div align="center">
  Built by <b>Sam (Islam) Elmenyawy</b> · AI/ML Engineer, Computer Vision ·
  <a href="https://www.linkedin.com/in/islam-ahmed-00188015a/">LinkedIn</a> ·
  <a href="https://github.com/eslamelmenyawy">GitHub</a>
</div>
