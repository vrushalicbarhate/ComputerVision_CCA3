# Input data

Put your own files here, with exactly these names:

| File | Used in | What to capture |
|---|---|---|
| `photo1.jpg` | Unit 3, Unit 4, Unit 5 | Your main photo, taken with your phone. Pick a scene with visible details and texture. If it includes a person's face (front-facing), the Unit 5 face detector has something to find. |
| `photo2.jpg` | Unit 4 | The **same** object/scene as `photo1.jpg`, taken from a slightly different angle or distance. |
| `objects.jpg` | Unit 3 (watershed) – *optional* | Coins, bottle caps or sweets **touching each other** on a plain white sheet of paper, photographed from above. If this file is missing, watershed uses `photo1.jpg`. |
| `video.mp4` | Unit 5 | A 15–30 second video with a person or object moving. Keep the phone still (rest it on something) for at least part of the video, and make sure the background has some detail. |

Tips:
* Do **not** edit or filter the originals.
* iPhone photos in `.HEIC` format cannot be read by OpenCV. Either convert them to `.jpg`, or set *Settings → Camera → Formats → Most Compatible* before shooting.
* If your video is `.mov`, either rename the path in the notebook (`VIDEO_PATH = "data/video.mov"`) or convert it to `.mp4`.
* Videos are **not** committed to Git (see `.gitignore`). Upload the original video to Google Drive and put the link in the README and the report.
