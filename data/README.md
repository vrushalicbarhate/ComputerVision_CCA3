# Input data

These are the input files used by the three notebooks. All photos were taken with my phone and are unedited. The video is a stock video, not my own recording (see the note below).

| File | Used in | What it shows |
|---|---|---|
| `photo1.jpeg` | Unit 3, Unit 4 | A courtyard in Goa with a wooden table and benches. It has clear edges and plenty of texture (wood, walls, leaves), which suits thresholding, Canny edge detection and SIFT keypoints. |
| `photo2.jpeg` | Unit 4 | The same scene as `photo1.jpeg`, taken from a slightly different angle. Used for SIFT feature matching. |
| `objects.jpeg` | Unit 3 (watershed) | 17 of my daily-use objects, often found in my bags. Used to compare a plain binary mask with marker-based watershed. If this file is missing, the notebook falls back to the main photo. |
| `photo5.jpeg` | Unit 5 (object recognition) | A photo of a person with everyday objects such as bottles, bowls and glasses. Used to test the Viola–Jones face detector and YOLO. |
| `video.mp4` | Unit 5 (motion estimation) | A stock video of people moving, downloaded from YouTube (allowed by the course instructor). 14.5 seconds, 30 fps. Used for frame differencing, GMM background subtraction and Lucas–Kanade optical flow, and for the video frames in the object-recognition test. |

Notes:

* The photos are the originals, with no editing or filters.
* The video is **not** committed to Git (see `.gitignore`). The original video and all other files are in the Google Drive folder:
  https://drive.google.com/drive/folders/19sJVs1wFgj5AbIVdJ_XmkazaDgCNh2IU?usp=sharing
* OpenCV cannot read `.HEIC` photos from iPhones. Convert them to `.jpg` or `.jpeg` first.
* The notebooks need OpenCV 4.x (`opencv-python>=4.8,<5`). OpenCV 5 moved the Haar cascade face detector out of the main package, so the Viola–Jones cell fails there.
