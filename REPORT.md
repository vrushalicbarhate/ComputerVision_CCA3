# Technical Report — CCA 3 Practical Assignment (Units 3, 4 and 5)

| | |
|---|---|
| **Student** | Vrushali Barhate |
| **Roll number** | 1272250540 |
| **Course** | Computer Vision – CCA 2 |
| **Git repository** | https://github.com/vrushalicbarhate/ComputerVision_CCA3 |
| **Google Drive folder** (original photo, video, outputs) | https://drive.google.com/drive/folders/19sJVs1wFgj5AbIVdJ_XmkazaDgCNh2IU?usp=sharing |

---

## 1. Objective

The objective is to implement and compare important computer-vision techniques from three units, using my own photographs and a stock video (downloaded from YouTube, which my instructor allowed):

* **Unit 3** – image restoration (noise removal), morphological processing and segmentation,
* **Unit 4** – feature extraction (HOG, SIFT) and image matching,
* **Unit 5** – object recognition (Viola–Jones, YOLO) and motion estimation (frame differencing, GMM, Lucas–Kanade).

All code is in Python using OpenCV, NumPy, Matplotlib, scikit-image and Ultralytics YOLO.

---

## 2. Input data

| File | Description | Device | Resolution / duration |
|---|---|---|---|
| `data/photo1.jpg` | Wooden table and benches in a courtyard, taken in Goa | OnePlus Nord CE4 | resized to 800 × 1066 px for processing |
| `data/photo2.jpg` | Same scene as photo 1 from a different angle | OnePlus Nord CE4 | resized to 800 × 1066 px for processing |
| `data/objects.jpg` | 17 of my daily-use objects, often found in my bags (used for watershed) | OnePlus Nord CE4 | resized to 800 px wide for processing |
| `data/video.mp4` | Stock video of people moving, downloaded from YouTube (allowed by the instructor) | – | 14.5 s, 30 fps, 434 frames (480 × 270 after resizing) |

The photographs are unedited. For processing, photos are resized to 800 pixels wide and video frames to 480 pixels on the longer side, to keep the computation fast.

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

**Requirement:** OpenCV 4.x (`opencv-python>=4.8,<5`). OpenCV 5 moved the Haar cascade face detector out of the main package, so `cv2.CascadeClassifier` is missing there and the Viola–Jones cell fails.

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

| Image | PSNR (dB) | SSIM |
|---|---|---|
| Gaussian noisy (no filter) | 20.46 | 0.434 |
| Gaussian noise → Gaussian filter | 25.73 | 0.734 |
| Gaussian noise → Median filter | 23.73 | 0.649 |
| S&P noisy (no filter) | 15.03 | 0.276 |
| S&P noise → Gaussian filter | 22.94 | 0.568 |
| S&P noise → Median filter | 24.37 | 0.774 |

**Observation:** Every filter improved both scores. For Gaussian noise, the Gaussian filter was better than the median filter on both measures (25.73 dB and SSIM 0.734, against 23.73 dB and 0.649), because averaging cancels random positive and negative errors. For salt-and-pepper noise, the median filter was better (24.37 dB and SSIM 0.774, against 22.94 dB and 0.568), because it discards the extreme 0 and 255 values, while the Gaussian filter spreads each black or white dot into its neighbours. The gap on salt-and-pepper noise is larger in SSIM (0.206) than in PSNR (1.43 dB), so the median filter preserves the structure of the image much better.

### 4.4 Thresholding

| ![](outputs/unit3/00_gray.jpg) | ![](outputs/unit3/03_binary_fixed.jpg) | ![](outputs/unit3/03_binary_otsu.jpg) |
|---|---|---|
| *Figure 6 – Grayscale* | *Fixed threshold (127)* | *Otsu threshold (112)* |

**Observation:** Otsu automatically chose a threshold of 112 from the histogram of this photo, which is 15 grey levels below the fixed value of 127, so the two masks are close. The advantage of Otsu is that it adapts to the lighting of each photo, while 127 is only a guess. My courtyard photo has dappled shadows on the walls and floor, so both masks are patchy in the shaded areas.

### 4.5 Morphological operations (5 × 5 square kernel)

| ![](outputs/unit3/04_erosion.jpg) | ![](outputs/unit3/04_dilation.jpg) | ![](outputs/unit3/04_opening.jpg) | ![](outputs/unit3/04_closing.jpg) |
|---|---|---|---|
| *Erosion* | *Dilation* | *Opening* | *Closing* |

*Figure 7 – Morphological results on the Otsu binary image.*

| Operation | White pixels (%) |
|---|---|
| Binary (Otsu) | 46.9 |
| Erosion | 36.5 |
| Dilation | 60.5 |
| Opening | 43.4 |
| Closing | 49.1 |

**Observation:** Erosion shrinks the white regions (46.9% to 36.5%) and dilation grows them (to 60.5%), so both change the size of objects a lot. Opening (erosion then dilation) removed small white specks and reduced the white area only slightly, to 43.4%. Closing (dilation then erosion) filled small black holes and raised it slightly, to 49.1%. Opening and closing therefore clean the mask while keeping the object size close to the original.

### 4.6 Canny edge detection

| ![](outputs/unit3/05_canny_no_blur.jpg) | ![](outputs/unit3/05_canny_low.jpg) |
|---|---|
| *No blur (100, 200)* | *Blurred, low thresholds (30, 100)* |
| ![](outputs/unit3/05_canny_high.jpg) | ![](outputs/unit3/05_canny_noisy.jpg) |
| *Blurred, high thresholds (100, 200)* | *On the Gaussian-noisy image (100, 200)* |

*Figure 8 – Canny edge maps.*

| Setting | Edge pixels (%) |
|---|---|
| No blur (100, 200) | 12.79 |
| Blurred, low (30, 100) | 13.37 |
| Blurred, high (100, 200) | 6.56 |
| Noisy image (100, 200) | 25.17 |

**Observation:** Blurring first and using the higher thresholds (100, 200) gave the fewest edge pixels (6.56%), because weak edges from leaf texture and shadows are suppressed and only strong outlines remain. Lower thresholds (30, 100) kept more weak edges (13.37%). On the Gaussian-noisy image, 25.17% of pixels became edges, nearly four times the blurred result, because every noisy pixel looks like an edge. This shows that noise should be removed before edge detection. The courtyard's leaves create many small edges in every setting.

### 4.7 Watershed segmentation

| ![](outputs/unit3/06_watershed_mask.jpg) | ![](outputs/unit3/06_watershed_distance.jpg) | ![](outputs/unit3/06_watershed_sure_fg.jpg) |
|---|---|---|
| *Binary mask* | *Distance transform* | *Sure foreground (markers)* |
| ![](outputs/unit3/06_watershed_regions.jpg) | ![](outputs/unit3/06_watershed_result.jpg) | |
| *Watershed regions* | *Boundaries (red)* | |

*Figure 9 – Marker-based watershed.*

**Result:** The photo contains 17 objects, counted by eye.

| Method | Objects counted |
|---|---|
| Plain binary mask, all blobs | 83 |
| Plain binary mask, ignoring blobs under 500 px | 31 |
| Watershed, default `FG_RATIO` = 0.3 | 3 |
| Watershed, after lowering `FG_RATIO` to 0.1 | 17 |

The plain mask over-counted because noise, shadows and patterns on the objects broke the picture into many extra pieces. With the default setting, watershed found only 3 objects, because the sure-foreground step keeps pixels above 50% of the single largest distance in the image. The biggest object sets that cutoff, and smaller objects fall below it and disappear. Lowering `FG_RATIO` keeps the centres of the smaller objects, and watershed then found 17 regions, matching the real count.

### 4.8 Comparison — Task 1

| Job | Techniques compared | Best on my images | Reason |
|---|---|---|---|
| Gaussian noise | Gaussian vs median filter | Gaussian filter | 25.73 dB / SSIM 0.734 against 23.73 dB / 0.649; averaging cancels random errors |
| Salt-and-pepper noise | Gaussian vs median filter | Median filter | 24.37 dB / SSIM 0.774 against 22.94 dB / 0.568; extreme values are discarded, not spread |
| Binary mask | Fixed vs Otsu | Otsu | Chose 112 from the histogram, so it adapts to the photo; the masks are close here |
| Removing specks / filling holes | Erosion, dilation, opening, closing | Opening and closing | They clean the mask and change the white area by only 3 to 4 points (46.9% to 43.4% and 49.1%), while erosion and dilation change it by 10 to 14 points |
| Edges | Canny settings | Blurred, (100, 200) | 6.56% edge pixels against 13.37% (low thresholds) and 25.17% (noisy image) |
| Touching objects | Plain mask vs watershed | Watershed (after tuning) | 17 objects found, matching my count; the plain mask gave 83 blobs (31 over 500 px) |

---

## 5. Task 2 — Feature extraction and matching (Unit 4)

### 5.1 HOG features

HOG settings: 9 orientation bins, 16 × 16 pixel cells, 2 × 2 cell blocks. Feature vector length: **114,660** values for each photo (1066 × 800 pixels). This comes from 66 × 50 cells, 65 × 49 = 3,185 overlapping blocks, and 4 cells × 9 bins = 36 values per block (3,185 × 36 = 114,660).

| ![](outputs/unit4/01_hog_photo1.jpg) | ![](outputs/unit4/01_hog_photo2.jpg) |
|---|---|
| *Figure 10 – HOG of photo 1* | *HOG of photo 2* |

### 5.2 SIFT keypoints

Photo 1: 8,474 keypoints, photo 2: 10,222 keypoints; each descriptor has 128 values. Circles show the keypoint scale and the line shows its orientation.

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

| Experiment | Keypoints (photo 1) | Keypoints (photo 2) | Good matches | RANSAC inliers | Inlier % |
|---|---|---|---|---|---|
| Photo 1 vs Photo 2 (original) | 8474 | 10222 | 208 | 105 | 50.5 |
| Photo 1 vs Photo 2 rotated 45° | 8474 | 9704 | 180 | 96 | 53.3 |
| Photo 1 vs Photo 2 resized 50% | 8474 | 2922 | 151 | 77 | 51.0 |
| Photo 1 vs Photo 2 rotated 90° + 50% | 8474 | 2932 | 145 | 81 | 55.9 |

**Observation:** SIFT kept finding correct matches after rotation and resizing. The inlier percentage was 50.5% for the original pair, 53.3% after a 45° rotation and 51.0% after resizing to 50%. Resizing reduced the keypoints of photo 2 from 10,222 to 2,922 (about 70%) because fine details are lost, but good matches fell by only about 27% (208 to 151). With a 90° rotation and a 50% resize it gave 145 good matches and 81 inliers (55.9%). The inlier percentage stayed in a narrow band of 50–56% in every case, so rotation and scale changes did not make the matching worse; the small differences between rows are within normal variation of RANSAC. This happens because each keypoint stores its own orientation and the detector searches over many scales. About half of the good matches are inliers, probably because the courtyard is not a flat surface (RANSAC fits one plane, but the table, wall and trees are at different depths) and the leaves are repetitive. HOG is different: it describes the image on a fixed grid, so rotating the image changes every cell. It is suited to detecting upright objects, not to matching points.

---

## 6. Task 3 — Object recognition: Viola–Jones vs YOLO (Unit 5)

Viola–Jones: OpenCV's pretrained `haarcascade_frontalface_default.xml` (scaleFactor 1.1, minNeighbors 5). YOLO: pretrained `yolov8n.pt` (COCO, 80 classes, confidence ≥ 0.25). The test photo is my courtyard photo, which contains no person, so it also works as a false-alarm test. The three video frames show a crowd of people.

| Haar cascade | YOLO |
|---|---|
| ![](outputs/unit5/01_haar_Photo.jpg) | ![](outputs/unit5/01_yolo_Photo.jpg) |
| ![](outputs/unit5/01_haar_Frame_25.jpg) | ![](outputs/unit5/01_yolo_Frame_25.jpg) |
| ![](outputs/unit5/01_haar_Frame_50.jpg) | ![](outputs/unit5/01_yolo_Frame_50.jpg) |
| ![](outputs/unit5/01_haar_Frame_75.jpg) | ![](outputs/unit5/01_yolo_Frame_75.jpg) |

*Figure 16 – Detections (left: Haar faces, right: YOLO boxes with class and confidence).*

| Image | Haar: faces found | Haar time (ms) | YOLO: objects (confidence) | YOLO time (ms) |
|---|---|---|---|---|
| Photo | 1 | 833.5 | bench (0.74), bench (0.61) | 148.4 |
| Frame 25% | 0 | 519.0 | 19 objects: 16 person (0.29–0.90), 2 handbag (0.63, 0.27), traffic light (0.43) | 195.5 |
| Frame 50% | 1 | 516.2 | 17 objects: 14 person (0.27–0.89), handbag (0.45), traffic light (0.31), bicycle (0.29) | 123.9 |
| Frame 75% | 1 | 504.8 | 15 objects: 11 person (0.28–0.89), 3 handbag (0.56, 0.30, 0.30), skateboard (0.28) | 121.6 |

### 6.1 Failure cases

| Haar | YOLO |
|---|---|
| ![](outputs/unit5/02_fail_haar_Rotated_90.jpg) | ![](outputs/unit5/02_fail_yolo_Rotated_90.jpg) |
| ![](outputs/unit5/02_fail_haar_Very_dark.jpg) | ![](outputs/unit5/02_fail_yolo_Very_dark.jpg) |
| ![](outputs/unit5/02_fail_haar_Very_small.jpg) | ![](outputs/unit5/02_fail_yolo_Very_small.jpg) |

*Figure 17 – Rotated, dark and small versions of the test image.*

| Test | Haar: faces found | YOLO: objects (confidence) |
|---|---|---|
| Rotated 90° | 1 | none |
| Very dark | 0 | bench (0.73), bench (0.60), bench (0.34), dining table (0.28) |
| Very small | 0 | bench (0.72), bench (0.70), bench (0.31) |

**Failure cases observed:** My test photo is the courtyard, and YOLO found no person in it, so Haar's single detection on the photo and on the rotated copy is probably a false positive on leaves or wood texture (see Figure 16). Haar's zero faces on the dark and small versions is therefore not a real miss, because there was no face to lose. On the crowd frames Haar found only 0 to 1 faces while YOLO found 15 to 19 objects, mostly people with confidence up to 0.90, so Haar misses most faces, probably because they are small, turned or seen from the side and the cascade only handles frontal faces. YOLO found nothing in the 90° rotated image, because it is trained on upright scenes. It still detected benches in the dark image (0.73 and 0.60) and in the 200-pixel image (0.72 and 0.70). Low-confidence YOLO detections such as the traffic light (0.43), bicycle (0.29) and skateboard (0.28) may be wrong.

### 6.2 Comparison — Task 3

| | Viola–Jones (Haar) | YOLO (v8n) |
|---|---|---|
| Classes | frontal face only | 80 COCO classes |
| Output | box only | box + class + confidence |
| Speed on my computer | 505–834 ms per image | 122–196 ms per image |
| Rotated / dark / small | 1 / 0 / 0 faces (the photo has no face) | none / benches found / benches found |
| False detections | 1 face on a photo with no person | low-confidence objects (traffic light 0.43, bicycle 0.29, skateboard 0.28), not verified |

On my computer YOLO was about 3 to 6 times faster than Haar.

---

## 7. Task 4 — Video motion estimation (Unit 5)

The video has 434 frames at 30 fps (14.5 s), resized to 480 × 270.

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

| Method | Output | Avg time per frame (ms) | Average result |
|---|---|---|---|
| Frame differencing | binary mask | 1.37 | 4.24% pixels moving |
| GMM background subtraction | binary mask | 15.43 | 4.54% pixels moving |
| Lucas–Kanade optical flow | motion vectors | 16.38 | 0.29 px motion per point |

For the studied frame pair, Lucas–Kanade tracked 300 points, and only 20 of them moved more than 1 pixel (average motion 0.31 px, maximum 16 px).

**Observation:** Frame differencing was the fastest at 1.37 ms per frame, about 11 times faster than GMM (15.43 ms) and Lucas–Kanade (16.38 ms). Both masks marked a similar share of pixels as moving on average (4.24% and 4.54%). In Figure 24, GMM starts at 0% and needs about 2 seconds to learn the background before it reaches the same level as frame differencing. After that, both masks stay between about 3% and 6%, and they rise together near the end of the video (from about 13 s), where frame differencing reaches 11.5% in the last frame. The optical-flow graph has isolated spikes (up to 1.48 px near 4.1 s) where frame differencing shows no matching jump, so they are probably caused by a few mistracked or fast-moving points and not by motion of the whole frame.

### 7.1 Comparison — Task 4

| | Frame differencing | GMM (MOG2) | Lucas–Kanade |
|---|---|---|---|
| Output | mask of changed pixels | mask of foreground | motion vectors |
| Object shape | outline only (hollow) | full silhouette | sparse points |
| Learning needed | no | yes (first ≈ 2 s) | no |
| Object stops | disappears | fades slowly | vectors ≈ 0 |
| Speed (my computer) | 1.37 ms | 15.43 ms | 16.38 ms |
| Result on my video | 4.24% pixels moving on average | 4.54% pixels moving on average | 0.29 px average motion per point |

---

## 8. Conclusion

In this assignment I implemented image restoration, segmentation, feature matching, object recognition and motion estimation on my own photographs and a stock video. For restoration, the Gaussian filter gave the best result on Gaussian noise (25.73 dB, SSIM 0.734, against 23.73 dB and 0.649 for the median filter), while the median filter was best on salt-and-pepper noise (24.37 dB, SSIM 0.774, against 22.94 dB and 0.568 for the Gaussian filter), because it discards the extreme black and white values. Otsu chose a threshold of 112, opening removed small white specks, and closing filled small holes. Canny worked best after blurring, with thresholds of (100, 200) giving 6.56% edge pixels, against 25.17% on the noisy image, which shows that noise should be removed before edge detection. On my object photo the plain binary mask counted 83 blobs (31 after ignoring small specks), while marker-based watershed found 17 objects after I lowered `FG_RATIO` to 0.1, matching the 17 I counted by eye. SIFT matching kept an inlier rate of 53.3% after a 45° rotation and 51.0% after resizing to 50%, confirming its invariance, while HOG describes shape on a fixed grid (114,660 values) and is not rotation-invariant.

For recognition, YOLO found 15 to 19 objects in each crowd frame, mostly people with confidence up to 0.90, while the Haar cascade found at most 1 face, and YOLO was also faster on my computer (122–196 ms against 505–834 ms). Haar gave a probable false positive on my courtyard photo, and YOLO found nothing in the 90° rotated image, so both detectors have limits. For motion, frame differencing was the fastest (1.37 ms per frame) but only shows object edges, GMM gave full silhouettes after a learning phase of about 2 seconds, and Lucas–Kanade is the only method that gives the direction and speed of motion. The main thing I learned is that settings such as `FG_RATIO` matter, and that results must be checked against the real image and not only against the numbers.

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
