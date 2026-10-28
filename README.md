# CCA 3 – Computer Vision Practical Assignment (Units 3, 4 and 5)

**Student:** Vrushali Barhate · **PRN:** 1272250540

| Link | |
|---|---|
| Git repository | https://github.com/vrushalicbarhate/ComputerVision_CCA3 |
| Google Drive | https://drive.google.com/drive/folders/19sJVs1wFgj5AbIVdJ_XmkazaDgCNh2IU?usp=sharing |

This project implements image restoration, segmentation, feature extraction, object recognition and motion estimation in Python, using **my own photographs and my own video**.

## What is in each notebook

| Notebook | Unit / Task | Techniques |
|---|---|---|
| [Unit3_Restoration_Segmentation.ipynb](Unit3_Restoration_Segmentation.ipynb) | Unit 3 – Task 1 | Gaussian and salt-and-pepper noise, Gaussian and median filters, PSNR and SSIM, fixed and Otsu thresholding, erosion, dilation, opening, closing, Canny edge detection, marker-based watershed |
| [Unit4_Feature_Extraction.ipynb](Unit4_Feature_Extraction.ipynb) | Unit 4 – Task 2 | HOG features, SIFT keypoints and descriptors, brute-force matching with Lowe's ratio test, RANSAC inliers, matching after rotation and resizing |
| [Unit5_Recognition_Motion.ipynb](Unit5_Recognition_Motion.ipynb) | Unit 5 – Tasks 3 and 4 | Viola–Jones Haar cascade face detection, YOLOv8 object detection, failure-case tests, frame differencing, GMM (MOG2) background subtraction, Lucas–Kanade optical flow |

Other files:
* [REPORT.md](REPORT.md) – the technical report (inputs, outputs, explanations, comparison tables, conclusion).
* [requirements.txt](requirements.txt) – Python packages.

## Folder structure

```
CompVision/
├── data/                      input files (see data/README.md)
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── objects.jpg            (optional, for watershed)
│   └── video.mp4              (not in Git – download from Google Drive)
├── outputs/                   created when the notebooks run
│   ├── unit3/
│   ├── unit4/
│   └── unit5/
├── Unit3_Restoration_Segmentation.ipynb
├── Unit4_Feature_Extraction.ipynb
├── Unit5_Recognition_Motion.ipynb
├── REPORT.md
├── notes.md
├── requirements.txt
└── README.md
```

## Getting the input data

1. The photos (`photo1.jpg`, `photo2.jpg`, `objects.jpg`) are included in the `data/` folder.
2. The video is too large for Git. Download `video.mp4` from the Google Drive link above and put it in `data/`.
3. To use your own data instead, follow the instructions in [data/README.md](data/README.md) and keep the same file names.

## Setup

Requires **Python 3.10 or newer**.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install the packages
pip install -r requirements.txt
```

> OpenCV is kept below version 5 because OpenCV 5 removed the Haar cascade face detector (`cv2.CascadeClassifier`) that Task 3 needs.

## Running

Open the project folder in VS Code (choose the `.venv` kernel) or start Jupyter:

```bash
jupyter notebook
```

Open each notebook and run **Kernel → Restart & Run All**. The notebooks are independent and can be run in any order.

* All result images are saved to `outputs/unit3`, `outputs/unit4` and `outputs/unit5`.
* Tables are printed in Markdown format so they can be pasted straight into the report.
* The first run of the Unit 5 notebook downloads the pretrained YOLO model `yolov8n.pt` (about 6 MB), so it needs an internet connection once.
* Unit 5 also writes `outputs/unit5/motion_comparison.mp4` (original, frame difference, GMM mask, optical flow side by side).

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Could not read data/photo1.jpg` | Check the file name and that it is in `data/`. HEIC photos must be converted to JPG. |
| `Could not read data/video.mp4` | Check the name/extension; change `VIDEO_PATH` in the Unit 5 notebook if your file is `.mov`. |
| `module 'cv2' has no attribute 'CascadeClassifier'` | You have OpenCV 5. Run `pip install "opencv-python<5"`. |
| Watershed does not separate objects | Use a photo of touching objects on a plain background (`data/objects.jpg`), or change `FG_RATIO` (0.3–0.7). |
| Optical-flow arrows are tiny | Change `T` in the Unit 5 notebook to a moment where the object moves fast. |
