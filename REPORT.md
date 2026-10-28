# Technical Report — CCA 3 Practical Assignment (Units 3, 4 and 5)

| | |
|---|---|
| **Student** | Vrushali Barhate |
| **PRN** | 1272250540 |
| **Course** | Computer Vision – CCA 3 |
| **Git repository** | _paste your GitHub URL here_ |
| **Google Drive folder** (original photo, video, outputs) | _paste the Drive folder link here_ |

> How to use this template: run all three notebooks first (the images below are loaded from `outputs/`), then replace every _italic placeholder_ and paste the tables that the notebooks print. To submit as PDF, open this file in VS Code and use a "Markdown PDF" extension, or run `pandoc REPORT.md -o REPORT.pdf`.

---

## 1. Objective

The objective is to implement and compare important computer-vision techniques from three units, using my own photograph and my own video:

* **Unit 3** – image restoration (noise removal), morphological processing and segmentation,
* **Unit 4** – feature extraction (HOG, SIFT) and image matching,
* **Unit 5** – object recognition (Viola–Jones, YOLO) and motion estimation (frame differencing, GMM, Lucas–Kanade).

All code is in Python using OpenCV, NumPy, Matplotlib, scikit-image and Ultralytics YOLO.

---

## 2. Input data

| File | Description | Device | Resolution / duration |
|---|---|---|---|
| `data/photo1.jpg` | this photo consists of a table, it was taken in goa| oneplus nord ce4 |
| `data/photo2.jpg` | same scene as photo 1 from a different angle | oneplus nord ce4 | _…_ |
| `data/objects.jpg` | these are my dailyuse objects (watershed) | oneplys nord ce4 | _…_ |
| `data/video.mp4` | people moving stock video, downloaded from youtube|

All originals are unedited. For processing, photos are resized to 800 pixels wide and video frames to 480 pixels on the longer side, to keep the computation fast.

| ![](data/photo1.jpg) | ![](data/photo2.jpg) |
|---|---|
| *Figure 1 – Original photograph 1* | *Figure 2 – Photograph 2 (same scene, different view)* |

| ![](outputs/unit5/00_input_Frame_25.jpg) | ![](outputs/unit5/00_input_Frame_50.jpg) | ![](outputs/unit5/00_input_Frame_75.jpg) |
|---|---|---|
| *Figure 3 – Video frame at 25%* | *Video frame at 50%* | *Video frame at 75%* |

---

## 3. Source code

| Notebook | Tasks |
|---|---|
| `Unit3_Restoration_Segmentation.ipynb` | Task 1 – restoration and segmentation |
| `Unit4_Feature_Extraction.ipynb` | Task 2 – HOG, SIFT, matching |
| `Unit5_Recognition_Motion.ipynb` | Task 3 – Viola–Jones and YOLO; Task 4 – motion estimation |

Setup and run instructions are in `README.md`. Every notebook can be run with *Restart & Run All*.

---

## 4. Task 1 — Image restoration and segmentation (Unit 3)

### 4.1 Adding noise

Gaussian noise (mean 0, standard deviation 25) was added to every pixel, and salt-and-pepper noise was added by setting 5% of pixels to black and 5% to white.

| ![](outputs/unit3/00_original.jpg) | ![](outputs/unit3/01_gaussian_noisy.jpg) | ![](outputs/unit3/01_saltpepper_noisy.jpg) |
|---|---|---|
| *Figure 4 – Original* | *Gaussian noise (σ = 25)* | *Salt-and-pepper noise (10%)* |

### 4.2 Restoration with Gaussian and median filters (5 × 5)

The Gaussian filter replaces each pixel with a weighted average of its neighbours. The median filter replaces it with the middle value of its sorted neighbours, which throws away extreme values.

| | Gaussian filter | Median filter |
|---|---|---|
| **Gaussian noise** | ![](outputs/unit3/02_gaussian_noise_gaussian_filter.jpg) | ![](outputs/unit3/02_gaussian_noise_median_filter.jpg) |
| **Salt-and-pepper noise** | ![](outputs/unit3/02_sp_noise_gaussian_filter.jpg) | ![](outputs/unit3/02_sp_noise_median_filter.jpg) |

*Figure 5 – Restored images.*

### 4.3 Quality measurement (PSNR and SSIM)

PSNR measures pixel error in decibels (higher is better); SSIM measures structural similarity from 0 to 1 (1 = identical).

_Paste the table printed by the notebook here:_

| Image | PSNR (dB) | SSIM |
|---|---|---|
| Gaussian noisy (no filter) | | |
| Gaussian noise → Gaussian filter | | |
| Gaussian noise → Median filter | | |
| S&P noisy (no filter) | | |
| S&P noise → Gaussian filter | | |
| S&P noise → Median filter | | |

**Observation:** _e.g. The median filter is clearly best for salt-and-pepper noise (PSNR … dB, SSIM …) because it discards the extreme 0/255 values. For Gaussian noise both filters give similar results (… dB vs … dB)._

### 4.4 Thresholding

| ![](outputs/unit3/00_gray.jpg) | ![](outputs/unit3/03_binary_fixed.jpg) | ![](outputs/unit3/03_binary_otsu.jpg) |
|---|---|---|
| *Figure 6 – Grayscale* | *Fixed threshold (127)* | *Otsu threshold (_value_)* |

**Observation:** _Otsu automatically chose a threshold of … , which separates … better than the fixed value 127._

### 4.5 Morphological operations (5 × 5 square kernel)

| ![](outputs/unit3/04_erosion.jpg) | ![](outputs/unit3/04_dilation.jpg) | ![](outputs/unit3/04_opening.jpg) | ![](outputs/unit3/04_closing.jpg) |
|---|---|---|---|
| *Erosion* | *Dilation* | *Opening* | *Closing* |

*Figure 7 – Morphological results on the Otsu binary image.*

_Paste the "white pixels (%)" table here._

**Observation:** _Erosion shrinks white regions and removes small specks; dilation grows them and fills holes. Opening removed … while keeping object size; closing filled …_

### 4.6 Canny edge detection

| ![](outputs/unit3/05_canny_no_blur.jpg) | ![](outputs/unit3/05_canny_low.jpg) |
|---|---|
| *No blur (100, 200)* | *Blurred, low thresholds (30, 100)* |
| ![](outputs/unit3/05_canny_high.jpg) | ![](outputs/unit3/05_canny_noisy.jpg) |
| *Blurred, high thresholds (100, 200)* | *On the Gaussian-noisy image (100, 200)* |

*Figure 8 – Canny edge maps.*

_Paste the "edge pixels (%)" table here._

**Observation:** _Blurring first and using higher thresholds gives the cleanest outlines. On the noisy image, … % of pixels became edges, which shows that denoising is needed before edge detection._

### 4.7 Watershed segmentation

| ![](outputs/unit3/06_watershed_mask.jpg) | ![](outputs/unit3/06_watershed_distance.jpg) | ![](outputs/unit3/06_watershed_sure_fg.jpg) |
|---|---|---|
| *Binary mask* | *Distance transform* | *Sure foreground (markers)* |
| ![](outputs/unit3/06_watershed_regions.jpg) | ![](outputs/unit3/06_watershed_result.jpg) | |
| *Watershed regions* | *Boundaries (red)* | |

*Figure 9 – Marker-based watershed.*

**Result:** plain binary mask counted _…_ objects; watershed found _…_ objects.

### 4.8 Comparison — Task 1

| Job | Techniques compared | Best on my images | Reason |
|---|---|---|---|
| Gaussian noise | Gaussian vs median filter | _…_ | _…_ |
| Salt-and-pepper noise | Gaussian vs median filter | _Median_ | _…_ |
| Binary mask | Fixed vs Otsu | _…_ | _…_ |
| Removing specks / filling holes | Erosion, dilation, opening, closing | _…_ | _…_ |
| Edges | Canny settings | _…_ | _…_ |
| Touching objects | Plain mask vs watershed | _Watershed_ | _…_ |

---

## 5. Task 2 — Feature extraction and matching (Unit 4)

### 5.1 HOG features

HOG settings: 9 orientation bins, 16 × 16 pixel cells, 2 × 2 cell blocks. Feature vector length: _…_ values.

| ![](outputs/unit4/01_hog_photo1.jpg) | ![](outputs/unit4/01_hog_photo2.jpg) |
|---|---|
| *Figure 10 – HOG of photo 1* | *HOG of photo 2* |

### 5.2 SIFT keypoints

Photo 1: _…_ keypoints, photo 2: _…_ keypoints; each descriptor has 128 values. Circles show the keypoint scale and the line shows its orientation.

| ![](outputs/unit4/02_sift_keypoints_photo1.jpg) | ![](outputs/unit4/02_sift_keypoints_photo2.jpg) |
|---|---|
| *Figure 11 – SIFT keypoints, photo 1* | *SIFT keypoints, photo 2* |

### 5.3 Matching and the rotation/resize experiment

Matching: brute-force (Euclidean distance) with k = 2, Lowe's ratio test (0.75), and RANSAC homography to count correct matches (inliers).

![](outputs/unit4/04_matches_0.jpg)
*Figure 12 – Photo 1 vs photo 2.*

![](outputs/unit4/04_matches_1.jpg)
*Figure 13 – Photo 1 vs photo 2 rotated 45°.*

![](outputs/unit4/04_matches_2.jpg)
*Figure 14 – Photo 1 vs photo 2 resized to 50%.*

![](outputs/unit4/04_matches_3.jpg)
*Figure 15 – Photo 1 vs photo 2 rotated 90° and resized to 50%.*

_Paste the matching table printed by the notebook here:_

| Experiment | Keypoints (photo 1) | Keypoints (photo 2) | Good matches | RANSAC inliers | Inlier % |
|---|---|---|---|---|---|
| | | | | | |

**Observation:** _SIFT kept a high inlier percentage after rotation (…%) and resizing (…%) because each keypoint has its own orientation and scale. Resizing reduced the number of keypoints from … to … because fine details are lost. HOG is not rotation-invariant, so it is suited to detection, not matching._

---

## 6. Task 3 — Object recognition: Viola–Jones vs YOLO (Unit 5)

Viola–Jones: OpenCV's pretrained `haarcascade_frontalface_default.xml` (scaleFactor 1.1, minNeighbors 5). YOLO: pretrained `yolov8n.pt` (COCO, 80 classes, confidence ≥ 0.25).

| Haar cascade | YOLO |
|---|---|
| ![](outputs/unit5/01_haar_Photo.jpg) | ![](outputs/unit5/01_yolo_Photo.jpg) |
| ![](outputs/unit5/01_haar_Frame_25.jpg) | ![](outputs/unit5/01_yolo_Frame_25.jpg) |
| ![](outputs/unit5/01_haar_Frame_50.jpg) | ![](outputs/unit5/01_yolo_Frame_50.jpg) |
| ![](outputs/unit5/01_haar_Frame_75.jpg) | ![](outputs/unit5/01_yolo_Frame_75.jpg) |

*Figure 16 – Detections (left: Haar faces, right: YOLO boxes with class and confidence).*

_Paste the detection table here:_

| Image | Haar: faces found | Haar time (ms) | YOLO: objects (confidence) | YOLO time (ms) |
|---|---|---|---|---|
| | | | | |

### 6.1 Failure cases

| Haar | YOLO |
|---|---|
| ![](outputs/unit5/02_fail_haar_Rotated_90.jpg) | ![](outputs/unit5/02_fail_yolo_Rotated_90.jpg) |
| ![](outputs/unit5/02_fail_haar_Very_dark.jpg) | ![](outputs/unit5/02_fail_yolo_Very_dark.jpg) |
| ![](outputs/unit5/02_fail_haar_Very_small.jpg) | ![](outputs/unit5/02_fail_yolo_Very_small.jpg) |

*Figure 17 – Rotated, dark and small versions of the test image.*

**Failure cases observed:** _e.g. Haar found 0 faces in the rotated image because it was trained only on upright faces, while YOLO still detected "person" (0.xx). Haar produced a false face on … . YOLO labelled … as … ._

### 6.2 Comparison — Task 3

| | Viola–Jones (Haar) | YOLO (v8n) |
|---|---|---|
| Classes | frontal face only | 80 COCO classes |
| Output | box only | box + class + confidence |
| Speed on my computer | _… ms_ | _… ms_ |
| Rotated / dark / small | _…_ | _…_ |
| False detections | _…_ | _…_ |

---

## 7. Task 4 — Video motion estimation (Unit 5)

| ![](outputs/unit5/10_consecutive_1.jpg) | ![](outputs/unit5/10_consecutive_2.jpg) | ![](outputs/unit5/10_consecutive_3.jpg) | ![](outputs/unit5/10_consecutive_4.jpg) |
|---|---|---|---|
| *Frame T* | *T + 1* | *T + 2* | *T + 3* |

*Figure 18 – Four consecutive frames.*

| ![](outputs/unit5/11_frame_difference_mask.jpg) | ![](outputs/unit5/12_gmm_mask.jpg) |
|---|---|
| *Figure 19 – Frame differencing mask (threshold 25)* | *Figure 20 – GMM (MOG2) foreground mask* |
| ![](outputs/unit5/12_gmm_background.jpg) | ![](outputs/unit5/13_optical_flow_arrows.jpg) |
| *Figure 21 – Background learned by MOG2* | *Figure 22 – Lucas–Kanade motion vectors (drawn 5× longer)* |

![](outputs/unit5/14_optical_flow_tracks.jpg)
*Figure 23 – Paths of points tracked over 20 frames.*

![](outputs/unit5/15_motion_over_time.png)
*Figure 24 – Foreground size (frame differencing vs GMM) and average optical-flow motion over the whole video.*

_Paste the comparison table printed by the notebook here:_

| Method | Output | Avg time per frame (ms) | Average result |
|---|---|---|---|
| Frame differencing | binary mask | | |
| GMM background subtraction | binary mask | | |
| Lucas–Kanade optical flow | motion vectors | | |

### 7.1 Comparison — Task 4

| | Frame differencing | GMM (MOG2) | Lucas–Kanade |
|---|---|---|---|
| Output | mask of changed pixels | mask of foreground | motion vectors |
| Object shape | outline only (hollow) | full silhouette | sparse points |
| Learning needed | no | yes (first _…_ s) | no |
| Object stops | disappears | fades slowly | vectors ≈ 0 |
| Speed (my computer) | _… ms_ | _… ms_ | _… ms_ |
| Result on my video | _…_ | _…_ | _…_ |

---

## 8. Conclusion

_Write 5–8 sentences in your own words, for example:_
_In this assignment I implemented … on my own photograph and video. For restoration, the median filter gave the best result on salt-and-pepper noise (PSNR … dB), while both filters were similar on Gaussian noise. Otsu thresholding with opening/closing produced clean masks, Canny worked best after blurring, and marker-based watershed separated … touching objects that a plain mask merged. SIFT matching remained reliable after rotation and resizing (inlier % …), confirming its invariance, while HOG described shape but is not rotation-invariant. YOLO detected more object types with confidence scores and handled rotated/dark images better than the Haar cascade, which was faster but missed … . For motion, frame differencing was fastest but only showed object edges, GMM gave full silhouettes after a short learning phase, and Lucas–Kanade gave the direction and speed of motion._

---

## 9. References

1. OpenCV documentation – https://docs.opencv.org/4.x/
2. scikit-image documentation – https://scikit-image.org/docs/stable/
3. Ultralytics YOLO documentation – https://docs.ultralytics.com/
4. N. Otsu, "A threshold selection method from gray-level histograms", 1979.
5. J. Canny, "A computational approach to edge detection", 1986.
6. Z. Wang et al., "Image quality assessment: from error visibility to structural similarity" (SSIM), 2004.
7. N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection", 2005.
8. D. Lowe, "Distinctive image features from scale-invariant keypoints" (SIFT), 2004.
9. P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features", 2001.
10. J. Redmon et al., "You Only Look Once: unified, real-time object detection", 2016.
11. Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction" (MOG2), 2004.
12. B. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision", 1981.
