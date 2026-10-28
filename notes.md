# CCA 3 — Study Notes (Units 3, 4 and 5)

These notes explain **everything** in this project in simple language.
Every technical word is explained the first time it is used.
Each section ends with **likely viva / interview questions** and short answers you can say out loud.

| Notebook | Unit | Task |
|---|---|---|
| `Unit3_Restoration_Segmentation.ipynb` | Unit 3 | Task 1 – noise, filters, PSNR/SSIM, thresholding, morphology, Canny, watershed |
| `Unit4_Feature_Extraction.ipynb` | Unit 4 | Task 2 – HOG, SIFT, feature matching, rotation/resize test |
| `Unit5_Recognition_Motion.ipynb` | Unit 5 | Task 3 – Viola–Jones vs YOLO; Task 4 – frame differencing, GMM, Lucas–Kanade |

## Contents

0. [Basics you must know first](#0-basics-you-must-know-first)
1. [Unit 3A — Image restoration](#1-unit-3a--image-restoration)
2. [Unit 3B — Thresholding, morphology, Canny and watershed](#2-unit-3b--thresholding-morphology-canny-and-watershed)
3. [Unit 4 — HOG, SIFT and feature matching](#3-unit-4--hog-sift-and-feature-matching)
4. [Unit 5A — Object recognition: Viola–Jones and YOLO](#4-unit-5a--object-recognition-violajones-and-yolo)
5. [Unit 5B — Motion estimation](#5-unit-5b--motion-estimation)
6. [One-page cheat sheet](#6-one-page-cheat-sheet)
7. [How the project is organised](#7-how-the-project-is-organised)
8. [Script for the 5–8 minute explanation video](#8-script-for-the-58-minute-explanation-video)
9. [Glossary](#9-glossary)

---

## 0. Basics you must know first

### 0.1 What is a digital image?

A digital image is a **grid of tiny squares called pixels** ("picture elements").
Each pixel stores a number that says how bright it is.

* In normal (8-bit) images the number goes from **0 = black** to **255 = white**.
* Why 255? 8 bits can store 2⁸ = 256 different values: 0, 1, 2 … 255.
* In Python an image is a **NumPy array** (a table of numbers). Its data type is `uint8` = "unsigned 8-bit integer" = whole numbers from 0 to 255.

A tiny 3 × 3 grayscale image (dark, with one bright pixel in the middle):

```
 10  10  10
 10 200  10
 10  10  10
```

### 0.2 Grayscale vs colour

* **Grayscale** – one number per pixel. Shape of the array: `(height, width)`.
* **Colour** – three numbers per pixel: Blue, Green, Red. Shape: `(height, width, 3)`. The 3 is the number of **channels**.
* **OpenCV stores colour as BGR** (blue first), while matplotlib expects **RGB**. That is why our `show()` function calls `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. If you forget it, red and blue swap and faces look blue.
* Converting colour to gray: `gray = 0.299·R + 0.587·G + 0.114·B`. Green gets the biggest weight because human eyes are most sensitive to green.
* Why use grayscale? Many algorithms (thresholding, edges, morphology, Haar, SIFT, optical flow) only need brightness. One channel = 3× less data = faster and simpler.

### 0.3 Image shape and why we resize

An image 800 pixels wide and 600 pixels tall has shape `(600, 800, 3)` – **rows (height) first, then columns (width)**.
That is 480,000 pixels × 3 channels = 1.44 million numbers.

A 12-megapixel phone photo has 36 million numbers, so every step would be slow. That is why each notebook shrinks the photo to 800 pixels wide (and video frames to 480 pixels) before processing.

### 0.4 Kernel, filter and convolution (the "sliding window")

Most image operations in this project use the same idea:

1. Take a small grid of numbers called a **kernel** (or filter, or window), e.g. 3 × 3 or 5 × 5.
2. Put it on top of the image, centred on one pixel.
3. Combine the pixels under the window (for example, average them) and write the answer into the centre pixel of the output.
4. Slide the window to the next pixel and repeat for the whole image.

The math name for "multiply each pixel by a kernel weight and add up" is **convolution**.

Example – a 3 × 3 **mean (average) filter** has all weights = 1/9:

```
 10  10  10
 10 200  10      →  new centre value = (8 × 10 + 200) / 9 = 31
 10  10  10
```

The bright spot is spread out into a weaker grey – that is "blurring".

* Kernel sizes are **odd** (3, 5, 7 …) so that there is a single centre pixel.
* A bigger kernel = stronger effect, but more blur and slower.

### 0.5 Gradient (edge strength)

The **gradient** tells you **how quickly the brightness changes** at a pixel, and **in which direction**.

* Flat area (a plain wall) → gradient ≈ 0.
* Edge (border of an object) → large gradient.

It is computed with small kernels called **Sobel filters**:
`Gx` = change from left to right, `Gy` = change from top to bottom.

* Magnitude (strength) = √(Gx² + Gy²)
* Direction (angle) = arctan(Gy / Gx)

Gradients are used by **Canny, HOG, SIFT and Lucas–Kanade**, so this is a key idea for the whole project.

### 0.6 Histogram

A histogram counts **how many pixels have each brightness value** (0 … 255).
A dark photo has most pixels on the left side; a bright photo on the right.
Otsu's thresholding uses the brightness histogram. HOG and SIFT use histograms of **edge directions**.

### 0.7 Binary image and mask

A **binary image** has only two values: 0 (black) and 255 (white).
A **mask** is a binary image that marks which pixels belong to something (white = "yes, this pixel is the object / is moving").

### 0.8 Libraries used

| Library | What we use it for |
|---|---|
| **OpenCV** (`cv2`) | reading images and video, filters, thresholding, morphology, Canny, watershed, SIFT, Haar cascades, background subtraction, optical flow, drawing |
| **NumPy** (`np`) | images are NumPy arrays; adding noise, counting pixels, maths |
| **Matplotlib** (`plt`) | showing images and charts inside the notebook |
| **scikit-image** (`skimage`) | PSNR, SSIM and HOG |
| **Ultralytics** | the pretrained YOLO object detector |

### 0.9 Interview questions — basics

1. **What is a pixel?** – The smallest element of a digital image; it stores a brightness value (or 3 colour values).
2. **Why are pixel values 0 to 255?** – They are stored in 8 bits, and 8 bits can hold 256 values.
3. **What is the shape of a colour image in NumPy?** – `(height, width, 3)`; height (rows) comes first.
4. **What is the difference between RGB and BGR?** – Only the order of the colour channels. OpenCV uses BGR for historical reasons; matplotlib uses RGB, so we convert before showing.
5. **What is a kernel? Why is its size odd?** – A small grid of weights that slides over the image. Odd size gives one exact centre pixel.
6. **What is convolution?** – Sliding a kernel over an image and, at each position, computing a weighted sum of the pixels under it.
7. **What is a gradient?** – The rate and direction of brightness change; large at edges.
8. **Why do we convert to grayscale?** – Most algorithms only need brightness; one channel is faster and simpler.
9. **What is `uint8` and why do we clip values?** – Whole numbers 0–255. If you add 10 to 250 in `uint8` it "wraps around" to 4 instead of 260. So we do the maths in float, then `np.clip(…, 0, 255)` and convert back.

---

## 1. Unit 3A — Image restoration

*(Notebook: `Unit3_Restoration_Segmentation.ipynb`, Part A)*

### 1.1 What is image restoration?

**Restoration** = undo damage to get back (close to) the **original** image.
It is different from **enhancement**, which just makes an image look nicer (for example, more contrast) – there is no "original" to go back to.

A simple model of damage:

```
noisy image  =  original image  +  noise
```

In our experiment we add the noise **ourselves**, so we **know** the clean original. That is why we can measure how good the restoration is with PSNR and SSIM (they need the original to compare with).

### 1.2 What is noise?

**Noise** = random, unwanted changes in pixel values.
Real causes: low light / high ISO, sensor heat, electrical interference, errors when sending the image, dead or stuck camera pixels.

### 1.3 Gaussian noise

* A random number is **added to every pixel**.
* The random numbers follow a **Gaussian (normal) distribution** – the "bell curve": most values are near 0, big values are rare.
* **Mean (μ) = 0** → on average the brightness does not change.
* **Standard deviation (σ) = 25** → how spread out the noise is. About 68% of noise values are between −25 and +25, and 95% between −50 and +50.
* It looks like fine grain all over the image (like a photo taken at night).

Code, line by line:

```python
np.random.seed(42)                                  # same random numbers every run → repeatable results
noise = np.random.normal(0, 25, original.shape)      # one random number per pixel per channel (mean 0, std 25)
gaussian_noisy = np.clip(original + noise, 0, 255).astype(np.uint8)
```

* `original + noise` gives decimal numbers, some below 0 or above 255.
* `np.clip(…, 0, 255)` → anything below 0 becomes 0, anything above 255 becomes 255.
* `.astype(np.uint8)` → back to a normal image type.

### 1.4 Salt-and-pepper noise (impulse noise)

* Only **some random pixels** are damaged, and they become **pure white (255, "salt")** or **pure black (0, "pepper")**. All other pixels are untouched.
* Real causes: dead/hot pixels, bit errors in transmission, dust.

```python
r = np.random.rand(original.shape[0], original.shape[1])  # random number 0..1 for every pixel
sp_noisy[r < 0.05] = 0      # 5% of pixels → black
sp_noisy[r > 0.95] = 255    # 5% of pixels → white
```

Total noise density = 10% of pixels.

### 1.5 Gaussian filter (Gaussian blur)

* Replaces each pixel with a **weighted average** of its neighbours.
* The weights follow a bell curve: the centre pixel gets the most weight, pixels further away get less.
* A typical 5 × 5 Gaussian kernel (divided by 273 so that the weights add up to 1):

```
1   4   7   4   1
4  16  26  16   4
7  26  41  26   7      ÷ 273
4  16  26  16   4
1   4   7   4   1
```

* `cv2.GaussianBlur(img, (5, 5), 0)` → kernel 5 × 5; the `0` means "OpenCV, calculate sigma (the spread) from the kernel size for me" (about 1.1 for 5 × 5).
* It is a **linear filter** (the output is a weighted sum of the inputs).

**Why it works on Gaussian noise:** the noise values are random + and −. When you average many neighbours, the + and − values cancel out.
**Downside:** it also averages pixels across edges, so edges become blurry.
**Why it is bad for salt-and-pepper:** one pixel of 255 among dark pixels pulls the average up, so the white dot becomes a grey smudge instead of disappearing.

### 1.6 Median filter

* Takes all the values under the window, **sorts** them and picks the **middle** value.
* Example (3 × 3 window with one salt and one pepper pixel):

```
values:  10 12 11 255 13 10 12 11 0
sorted:  0 10 10 11 11 12 12 13 255
median = 11          (the mean would be 37 → a visible grey smudge)
```

The extreme values 0 and 255 end up at the ends of the sorted list, so they are **never chosen**. That is why the median filter removes salt-and-pepper noise almost perfectly.

* `cv2.medianBlur(img, 5)` → 5 × 5 window = 25 values, the median is the 13th.
* It is a **non-linear filter** (sorting cannot be written as a weighted sum).
* It **keeps edges sharper** than the Gaussian filter: at an edge, the median picks a value from one side of the edge instead of inventing an in-between grey.
* Downsides: slower (sorting), and a big window removes thin lines and fine details.

### 1.7 Which filter for which noise?

| Noise | Best filter | Why |
|---|---|---|
| Gaussian (every pixel a bit wrong) | **both work, scores are close** (Gaussian is the textbook choice) | averaging cancels the small + / − errors; the median keeps edges sharper, so it sometimes scores a little higher |
| Salt-and-pepper (few pixels totally wrong) | **Median filter** | sorting throws away the extreme 0 / 255 values |

### 1.8 PSNR — Peak Signal-to-Noise Ratio

First compute the **MSE (Mean Squared Error)**: for every pixel, subtract restored from original, square it (so negative errors count too), and take the average.

```
MSE  = average of (original − restored)²
PSNR = 10 · log10( 255² / MSE )         unit: decibels (dB)
```

* 255 is the maximum possible pixel value ("peak").
* **Higher PSNR = better** (closer to the original). If the images are identical, MSE = 0 and PSNR is infinite.
* Why a logarithm? MSE values vary enormously; the log squeezes them into an easy range, and dB is the standard unit in signal processing.
* Fun fact: **+3 dB ≈ the error is halved** (10 · log10(2) ≈ 3).
* With σ = 25 Gaussian noise we can even *predict* the noisy PSNR: MSE ≈ 25² = 625, so PSNR ≈ 10 · log10(65025 / 625) ≈ **20.2 dB**.

| PSNR | Quality (rough guide) |
|---|---|
| below 20 dB | poor, very visible damage |
| 20 – 30 dB | acceptable |
| 30 – 40 dB | good |
| above 40 dB | almost identical |

**Weakness:** PSNR compares pixel by pixel and does not match how humans see. Making a photo slightly brighter by 10 levels gives MSE = 100 → PSNR ≈ 28 dB, even though it still looks perfect.

### 1.9 SSIM — Structural Similarity Index

SSIM compares images more like a **human** does. In small windows it compares three things:

1. **Luminance** – is the average brightness the same?
2. **Contrast** – is the spread of brightness (how much it varies) the same?
3. **Structure** – do the patterns (edges, textures) line up?

```
SSIM(x, y) = [(2·μx·μy + C1) · (2·σxy + C2)] / [(μx² + μy² + C1) · (σx² + σy² + C2)]
```

μ = mean, σ² = variance, σxy = how x and y vary together; C1, C2 are tiny constants that stop division by zero.

* Range: **1 = identical**, 0 = no similarity (it can go slightly below 0 in theory).
* The final SSIM is the average over all windows.
* In the code: `ssim(original, img, data_range=255, channel_axis=2)` → `data_range=255` because pixels go 0–255; `channel_axis=2` = "the colour channels are in the 3rd dimension, compute SSIM per channel and average".

| | PSNR | SSIM |
|---|---|---|
| Measures | pixel error | structure, brightness, contrast |
| Unit / range | dB, 0 … ∞ | 0 … 1 |
| Matches human eyes? | not well | much better |
| Speed | very fast | slower |

Both are **full-reference** metrics: they need the clean original. In real life you often do not have it; then **no-reference** metrics such as BRISQUE or NIQE are used.

### 1.10 What results to expect

Your numbers depend on your photo, but the pattern is always similar:

* Gaussian noisy: about 20 dB PSNR. After Gaussian or median filter: several dB higher (often 27–30 dB), SSIM much higher. The two filters are close; either can win.
* Salt-and-pepper noisy: about 15 dB. Gaussian filter helps only partly. **Median filter gives a big jump** (often 30+ dB, SSIM above 0.9).

Example from a test run (800 × 800 photo):

| Image | PSNR (dB) | SSIM |
|---|---|---|
| Gaussian noisy | 20.85 | 0.277 |
| → Gaussian filter | 29.55 | 0.682 |
| → Median filter | 29.91 | 0.750 |
| S&P noisy | 14.51 | 0.178 |
| → Gaussian filter | 24.02 | 0.492 |
| → Median filter | **34.16** | **0.962** |

Notice: SSIM shows the difference much more clearly than PSNR – the Gaussian filter leaves grey smudges that look bad to a human, and SSIM punishes that.

### 1.11 Interview questions — restoration

1. **Restoration vs enhancement?** – Restoration recovers the original from a damaged image (it has a model of the damage). Enhancement just improves the look; there is no "true" image to compare with.
2. **What is Gaussian noise? What does σ mean?** – Random values from a bell curve added to every pixel. σ (standard deviation) is the noise strength.
3. **What is salt-and-pepper noise?** – Random pixels set to pure white or black; caused by faulty pixels or transmission errors.
4. **Why is the median filter best for salt-and-pepper noise?** – The extreme values end up at the ends of the sorted list and are never chosen as the median, so they are removed completely.
5. **Why is the median filter called non-linear?** – Its output comes from sorting, not from a weighted sum of inputs.
6. **Gaussian filter vs mean (box) filter?** – The mean filter gives every neighbour equal weight; the Gaussian gives closer pixels more weight, so it blurs more naturally.
7. **What happens if you increase the kernel size?** – More noise is removed, but the image becomes blurrier and loses detail; processing is slower.
8. **What is PSNR? Formula? Unit?** – 10·log10(255²/MSE), in decibels; higher is better.
9. **What if MSE is zero?** – The images are identical, PSNR is infinite.
10. **What is SSIM and why is it better than PSNR?** – It compares luminance, contrast and structure in small windows, which matches human perception better. Range 0 to 1.
11. **Why do we need the original image for PSNR/SSIM?** – They are full-reference metrics – they measure the difference from a known clean image.
12. **Why did you set a random seed?** – So the noise is the same every time and results are repeatable.
13. **Other denoising methods?** – Bilateral filter (smooths but keeps edges), Non-Local Means (`cv2.fastNlMeansDenoisingColored`), Wiener filter, and deep-learning denoisers.
14. **Does the median filter work on Gaussian noise?** – Yes, partly. It reduces the noise and keeps edges, but averaging (Gaussian) is the theoretically better match for Gaussian noise.

---

## 2. Unit 3B — Thresholding, morphology, Canny and watershed

*(Notebook: `Unit3_Restoration_Segmentation.ipynb`, Part B)*

### 2.1 What is segmentation?

**Segmentation** = dividing an image into meaningful regions – deciding **which pixels belong to which object** (or to the background).
Main families: threshold-based (Otsu), edge-based (Canny), region-based (watershed), and deep-learning based (U-Net, Mask R-CNN, SAM).

### 2.2 Thresholding

Rule: **if pixel > T → white (255), else → black (0).** T is the threshold.

```python
_, binary_fixed = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
otsu_value, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

* `cv2.threshold` returns **two** things: the threshold it used, and the binary image. `_` means "I don't need this value".
* `THRESH_BINARY` → bright = white. `THRESH_BINARY_INV` would do the opposite.
* We **blur first** so that tiny noise does not turn into random white/black dots.

**Fixed threshold (127):** simple, but if the photo is dark or bright overall, 127 may be a bad choice.

**Otsu's method (automatic):**
1. Try every threshold from 0 to 255.
2. For each one, split pixels into a "dark group" and a "bright group".
3. Measure how spread out each group is (variance).
4. Pick the threshold that makes the two groups **most compact and most separated** (minimum within-group variance = maximum between-group variance).

Otsu works best when the histogram has **two clear peaks** (bimodal) – e.g. a dark object on a light background. When `THRESH_OTSU` is used, the threshold value we pass (0) is ignored.

**Adaptive thresholding** (not required, good to know): `cv2.adaptiveThreshold` uses a different threshold for each small area, which helps with uneven lighting (shadows on a page).

### 2.3 Morphological operations

"Morphology" means "the study of shape". These operations change the shape of white regions in a binary image.

They use a **structuring element** (kernel) – a small shape such as a 5 × 5 square: `np.ones((5, 5), np.uint8)`. (Other shapes – cross, ellipse – can be made with `cv2.getStructuringElement`.)

* **Erosion** – a pixel stays white **only if all** pixels under the kernel are white. White regions **shrink**; small white dots and thin lines disappear; touching objects can separate.
* **Dilation** – a pixel becomes white **if any** pixel under the kernel is white. White regions **grow**; small holes fill; broken parts join.
* **Opening = erosion, then dilation** – removes small white noise dots; big objects come back to (almost) their original size.
* **Closing = dilation, then erosion** – fills small black holes and gaps inside objects; big objects keep their size.

Example with a 3 × 3 kernel (1 = white). There is a 3 × 3 white square and one lonely white noise dot on the right:

```
Original           Erosion            Dilation           Opening
0 0 0 0 0 0 0      0 0 0 0 0 0 0      1 1 1 1 1 0 0      0 0 0 0 0 0 0
0 1 1 1 0 0 0      0 0 0 0 0 0 0      1 1 1 1 1 1 1      0 1 1 1 0 0 0
0 1 1 1 0 0 1      0 0 1 0 0 0 0      1 1 1 1 1 1 1      0 1 1 1 0 0 0
0 1 1 1 0 0 0      0 0 0 0 0 0 0      1 1 1 1 1 1 1      0 1 1 1 0 0 0
0 0 0 0 0 0 0      0 0 0 0 0 0 0      1 1 1 1 1 0 0      0 0 0 0 0 0 0
```

* Erosion: only the centre of the square survives; the noise dot is gone.
* Dilation: the square grows to 5 × 5 and the dot grows into a block.
* Opening: the square comes back to its original size, and the noise dot is **gone for good**.

Memory trick:
* **Opening** opens up the space around objects → **removes white specks outside** objects.
* **Closing** closes holes → **fills black specks inside** objects.

Useful facts:
* Opening never adds white pixels; closing never removes white pixels. So in the notebook's "white pixels %" table: erosion ≤ opening ≤ original ≤ closing ≤ dilation.
* Applying opening twice gives the same result as once (same for closing).
* `iterations=2` = repeat the operation twice (stronger effect).
* Other operations: **morphological gradient** = dilation − erosion (gives the outline), top-hat, black-hat.

### 2.4 Canny edge detection

John Canny (1986) designed it to find edges that are **thin, connected and not noisy**. Steps:

1. **Smooth** the image with a Gaussian blur to reduce noise.
   (OpenCV's `cv2.Canny` does **not** blur for you, so we blur first.)
2. **Gradient** – use Sobel filters to get edge strength and direction at every pixel.
3. **Non-maximum suppression** – make edges 1 pixel thin: keep a pixel only if it is stronger than its two neighbours across the edge.
4. **Double threshold** –
   above **high** → strong edge (keep);
   below **low** → not an edge (remove);
   between → weak edge (decide in step 5).
5. **Hysteresis** – keep a weak edge **only if it is connected** to a strong edge. This keeps real edges continuous and throws away isolated noise.

`cv2.Canny(image, low, high)` – Canny recommended high : low ≈ 2 : 1 or 3 : 1.

What the comparison in the notebook shows:

| Setting | Result |
|---|---|
| No blur | many tiny edges from texture and noise |
| Blurred, low thresholds (30, 100) | lots of detail, some noise |
| Blurred, high thresholds (100, 200) | only the main outlines – cleanest |
| Noisy image | edges everywhere – noise looks like edges → shows why we restore/denoise **before** analysing |

**Why Canny instead of plain Sobel?** Sobel gives thick, broken edges and keeps noise; Canny gives thin, connected edges with fewer false ones.

### 2.5 Watershed segmentation

**Goal:** separate objects that **touch** each other (e.g. coins in a pile) – plain thresholding sees them as one big blob.

**The idea (a landscape with water):**
Think of the image as a landscape: dark pixels are valleys, bright pixels are hills.
Pour water into every valley. As the water rises, water from two different valleys eventually meets on a ridge – build a **dam (boundary)** there. When everything is flooded, the dams are the segmentation lines.

**Problem:** every tiny dip in a real image becomes its own valley → hundreds of tiny pieces (**over-segmentation**).
**Solution:** **marker-controlled watershed** – we tell it exactly where the objects and the background are, and water only starts from those markers.

**Distance transform** (the trick that finds object centres):
For every white pixel, compute the distance to the nearest black pixel. The centre of a coin is far from the edge → big value; pixels near the edge → small value.
Two touching coins → **two peaks** with a dip where they touch. Keep only pixels above 50% of the maximum distance → **two separate blobs** → two markers.

Steps in our code:

| Step | Code | Meaning |
|---|---|---|
| 1 | `cv2.threshold(… THRESH_OTSU)` then flip if needed | binary mask, objects white. If more than half the image is white (`np.mean(mask) > 127`), the background came out white, so we invert it. |
| 2 | `MORPH_OPEN` | remove small noise |
| 3 | `cv2.dilate` → `sure_bg` | grow the objects a little; anything still black is **surely background** |
| 4 | `cv2.distanceTransform` + threshold at `FG_RATIO × max` → `sure_fg` | **surely foreground** – object centres |
| 5 | `unknown = sure_bg − sure_fg` | the border region we are not sure about |
| 6 | `cv2.connectedComponents(sure_fg)` | give each centre blob its own number. We add 1 so background = 1 and objects = 2, 3, 4 …; unknown pixels = 0 ("watershed, please decide") |
| 7 | `cv2.watershed(img, markers)` | floods from the markers; every pixel gets its region number, boundaries get **−1** (drawn in red) |

* `cv2.watershed` needs a 3-channel colour image and integer markers.
* Tuning `FG_RATIO`: too **high** → small objects lose their marker and are missed; too **low** → two touching objects get one marker and stay merged.
* The notebook prints "objects counted from the plain mask" vs "objects found by watershed" – watershed should count more because it splits touching objects.
* Limitations: needs good markers; shadows, reflections or a patterned background break the thresholding step.

### 2.6 Interview questions — morphology and segmentation

1. **What is segmentation?** – Dividing an image into meaningful regions, e.g. object vs background.
2. **What is thresholding?** – Turning a grayscale image into black and white using a cut-off value.
3. **How does Otsu choose the threshold?** – It tries all values and picks the one that best separates the dark and bright groups (maximum between-class variance). Works best with a two-peak histogram.
4. **When does a global threshold fail?** – With uneven lighting or shadows; then adaptive thresholding is better.
5. **What is a structuring element?** – The small shape (kernel) used by morphological operations.
6. **Erosion vs dilation?** – Erosion shrinks white regions (pixel stays white only if all neighbours are white); dilation grows them (white if any neighbour is white).
7. **Opening vs closing?** – Opening = erosion then dilation, removes small white noise. Closing = dilation then erosion, fills small black holes.
8. **Why not just use erosion to remove noise?** – Erosion also shrinks the real objects. Opening removes the noise and then grows the objects back.
9. **What is the morphological gradient?** – Dilation minus erosion; it gives the object outline.
10. **What are the steps of Canny?** – Gaussian smoothing, gradient (Sobel), non-maximum suppression, double threshold, hysteresis.
11. **Why two thresholds in Canny?** – Strong edges are kept for sure; weak edges are kept only if connected to strong ones. This avoids broken edges and removes isolated noise.
12. **What is non-maximum suppression?** – Keeping only the local maximum across an edge so edges are 1 pixel thin.
13. **What happens if Canny thresholds are too low / too high?** – Too low: too many edges, noise. Too high: real edges go missing.
14. **Explain watershed in simple words.** – The image is a landscape; flood it from markers; where two floods meet, draw a boundary.
15. **What is over-segmentation and how do you avoid it?** – Too many small regions because every small dip is a valley. Avoid it with markers (marker-controlled watershed).
16. **What does the distance transform do?** – Gives every white pixel its distance to the nearest black pixel; object centres get the highest values, which we use as markers.
17. **What does −1 mean in the watershed output?** – Boundary pixels between regions.

---

## 3. Unit 4 — HOG, SIFT and feature matching

*(Notebook: `Unit4_Feature_Extraction.ipynb`)*

### 3.1 What is a feature?

A **feature** is a set of numbers that describes an image (or part of it), so that a computer can **compare** images.
Good features are:
* **distinctive** – different things give different numbers,
* **repeatable** – the same thing gives the same numbers in another photo,
* **invariant / robust** – they do not change much with rotation, size, lighting,
* **compact** – not too many numbers.

Two kinds:
* **Global descriptor** – one vector for a whole image/window (e.g. **HOG**).
* **Local features** – many interesting points, each with its own description (e.g. **SIFT**).

**Detector vs descriptor:** the *detector* finds **where** the interesting points are; the *descriptor* describes **what the area around each point looks like**.

### 3.2 HOG — Histogram of Oriented Gradients

Invented by Dalal and Triggs (2005) for **detecting pedestrians**, together with an SVM classifier.

**Idea:** the shape of an object can be described by **the directions of its edges**, even without knowing their exact positions. A person has strong vertical edges on the sides, a round head, etc.

Steps (with our settings):

1. **Gradients** – for every pixel compute edge strength and direction. HOG uses angles 0–180° ("unsigned": dark→light and light→dark count the same).
2. **Cells** – split the image into cells of 16 × 16 pixels (`pixels_per_cell=(16, 16)`).
3. **Histogram per cell** – 9 direction bins (`orientations=9`), each 20° wide: 0°, 20°, … 160°. Every pixel **votes** for its direction, and the vote is weighted by edge strength (strong edges count more).
4. **Block normalisation** – group 2 × 2 cells into a block (`cells_per_block=(2, 2)`) and normalise it (scale the numbers so their total size is 1). This makes HOG robust to lighting and contrast changes. Blocks **overlap** (they slide by one cell).
5. **Concatenate** all block histograms into one long **feature vector**.

**Feature vector length:**

```
length = (number of blocks across) × (number of blocks down) × (2 × 2 cells × 9 bins = 36)
```

Example: an 800 × 600 photo → 50 × 37 cells → 49 × 36 blocks = 1,764 blocks × 36 = **63,504 numbers**.
(The original pedestrian detector used a 64 × 128 window with 8 × 8 cells → 7 × 15 blocks × 36 = 3,780 numbers.)

**The HOG picture:** each cell shows a small star of lines. A brighter/longer line = stronger edges in that direction. The lines follow the edges, so the HOG picture looks like a rough sketch of the object's outline.
`exposure.rescale_intensity(…)` just makes the very dark HOG picture brighter so we can see it.

**Uses:** pedestrian detection (OpenCV even has a ready-made HOG people detector), vehicle detection, handwritten character recognition, input to classifiers like SVM.

**Limitations:** **not rotation-invariant** (rotate the image → every cell's edge directions change), not scale-invariant (fixed cell size), long vector.

**Effect of settings:** smaller cells → more detail, longer vector; more bins → finer angles; bigger blocks → more robust to light but less local detail.

### 3.3 SIFT — Scale-Invariant Feature Transform

By David Lowe (1999, 2004). Its patent expired in March 2020, so it is now in the main OpenCV package as `cv2.SIFT_create()` (before, it was only in the "contrib" package).

SIFT finds **keypoints** that can be found again even after **rotation, resizing and lighting changes**. Four steps:

1. **Scale-space extrema detection (finding candidate points at all sizes)**
   * Blur the image again and again with stronger Gaussian blur.
   * Subtract each blurred image from the next one → **Difference of Gaussians (DoG)**. DoG highlights "blobs" of a certain size.
   * Do this at several image sizes (**octaves**: full size, half, quarter …).
   * A pixel is a candidate if it is bigger or smaller than **all 26 neighbours** (8 around it + 9 in the scale above + 9 in the scale below).
   * Because we search over many sizes, each feature is found at its "natural" size → **scale invariance**.
2. **Keypoint localisation** – refine the exact position; throw away weak (low-contrast) points and points lying on straight edges (they are unstable because they can slide along the edge).
3. **Orientation assignment** – make a histogram of gradient directions around the point (36 bins); the highest peak becomes the keypoint's **orientation**. The descriptor is then measured **relative to this direction** → **rotation invariance**.
4. **Descriptor** – take a 16 × 16 area around the keypoint (rotated to its orientation and scaled to its size), split it into 4 × 4 sub-squares, make an 8-bin direction histogram in each:
   **4 × 4 × 8 = 128 numbers**. The vector is normalised → robust to lighting changes.

What OpenCV gives for each keypoint: `pt` (x, y position), `size` (diameter), `angle` (orientation in degrees), `response` (strength), `octave`.
`des1.shape` = `(number of keypoints, 128)`.

`cv2.drawKeypoints(…, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)` draws a **circle** (size = scale) with a **line** (orientation) for every keypoint.

* **Invariant to:** scale, rotation, moderate lighting change, small viewpoint changes.
* **Struggles with:** big viewpoint changes, strong blur, plain surfaces with no texture (a white wall has no keypoints), repeated patterns (many look-alike points).
* **Alternatives:** SURF (faster, patented), **ORB** (free, very fast, binary descriptor, compared with Hamming distance), AKAZE, BRISK, and deep-learning features.

### 3.4 Feature matching

**Brute-Force matcher:** compare **every** descriptor in photo A with **every** descriptor in photo B.
`cv2.BFMatcher(cv2.NORM_L2)` – the distance is the **Euclidean (straight-line) distance** between the two 128-number vectors:
`distance = √((a₁−b₁)² + (a₂−b₂)² + … + (a₁₂₈−b₁₂₈)²)`. Small distance = similar.
(For binary descriptors like ORB you use `NORM_HAMMING`. For very many points, **FLANN** – a fast *approximate* nearest-neighbour matcher – is quicker.)

**knnMatch with k = 2:** for each point in A, return the **two** closest points in B.
Each match is a `DMatch` object: `queryIdx` (index in A), `trainIdx` (index in B), `distance`.

**Lowe's ratio test:** keep the match only if `best.distance < 0.75 × second_best.distance`.
*Why?* If a point has two almost equally good candidates (e.g. one window of a building among many identical windows), we cannot be sure which is correct → throw it away. Lowe's paper (using 0.8) showed this removes about 90% of wrong matches while losing less than 5% of correct ones. Values 0.7–0.8 are common.

**RANSAC + homography (checking which matches are really correct):**
* A **homography** is a 3 × 3 matrix that maps points in one image to points in another view of the same flat scene (it handles rotation, scale, shift and perspective). It needs at least **4** point pairs.
* **RANSAC (Random Sample Consensus):** repeat many times – pick 4 random matches, compute a homography, count how many of the other matches agree with it (land within 5 pixels). Keep the homography with the most agreement.
* Matches that agree = **inliers** (very likely correct). The rest = **outliers** (wrong matches).
* **Inlier % = inliers ÷ good matches × 100** – our main "matching quality" number.
* Note: a homography is exact only for flat scenes (or when the camera just rotates). For a 3D scene from a very different angle, some correct matches may be counted as outliers – fine for comparing our experiments.

`cv2.drawMatches(…, good[:50], …, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)` draws lines between the 50 best matches and hides unmatched points.

### 3.5 The rotation and resize experiment — what to expect

| Experiment | What usually happens | Why |
|---|---|---|
| Photo 1 vs Photo 2 | many good matches | same scene, slightly different view |
| Photo 2 rotated 45° | still many matches, high inlier % | each keypoint has its own orientation → rotation invariance. Some points are lost at the cut-off corners. |
| Photo 2 resized 50% | fewer keypoints and matches, but inlier % stays high | a smaller image has fewer fine details, so small-scale keypoints disappear; the remaining ones still match thanks to scale-space search |
| Rotated 90° + 50% | still works | SIFT handles both changes together |

If you tried the same with HOG, the feature vector would change a lot – HOG is not rotation-invariant.

### 3.6 HOG vs SIFT

| | HOG | SIFT |
|---|---|---|
| Type | global (dense grid over the whole window) | local (only at keypoints) |
| Output | one long vector | N keypoints × 128 numbers |
| Rotation invariant? | no | yes |
| Scale invariant? | no | yes |
| Main use | **detecting** objects of a known shape (pedestrians), with an SVM | **matching** images, panoramas, 3D reconstruction, object recognition |
| Speed | fast | slower |

### 3.7 Interview questions — features

1. **What is a feature in computer vision?** – Numbers describing an image or a region so it can be compared with others.
2. **Detector vs descriptor?** – The detector finds where the interesting points are; the descriptor describes their surroundings with numbers.
3. **What does HOG stand for and how does it work?** – Histogram of Oriented Gradients: compute gradients, make direction histograms in small cells, normalise over blocks, join everything into one vector.
4. **Why is block normalisation used in HOG?** – To make the features robust to lighting and contrast changes.
5. **Why does HOG use 9 bins?** – 0–180° split into 20° bins; Dalal and Triggs found 9 worked best.
6. **How do you calculate the HOG feature length?** – blocks across × blocks down × cells per block × bins.
7. **Is HOG rotation invariant?** – No.
8. **What does SIFT stand for? Why "scale-invariant"?** – Scale-Invariant Feature Transform. It searches for keypoints over many blur levels and image sizes, so the same point is found whether the object is big or small.
9. **How does SIFT achieve rotation invariance?** – It gives each keypoint a main orientation and computes the descriptor relative to it.
10. **What is DoG?** – Difference of Gaussians: subtracting two blurred versions of an image; it finds blobs and approximates the Laplacian of Gaussian.
11. **Why is the SIFT descriptor 128 numbers long?** – 4 × 4 sub-regions × 8 orientation bins.
12. **What is Lowe's ratio test?** – Keep a match only if the best distance is clearly smaller (e.g. < 0.75×) than the second best; removes ambiguous matches.
13. **What is RANSAC?** – An algorithm that finds the model (here a homography) agreed on by most matches, by repeatedly testing random small samples; it separates inliers from outliers.
14. **What is a homography? How many points does it need?** – A 3 × 3 transform between two views of a plane; at least 4 point pairs.
15. **Brute-force vs FLANN?** – Brute force checks every pair (exact, slow for big sets); FLANN is approximate but much faster.
16. **SIFT vs ORB?** – SIFT: float 128-D descriptor, more accurate, slower. ORB: binary descriptor, very fast, free, uses Hamming distance.
17. **Why did matches drop after resizing?** – The smaller image has fewer details, so fewer keypoints are detected.

---

## 4. Unit 5A — Object recognition: Viola–Jones and YOLO

*(Notebook: `Unit5_Recognition_Motion.ipynb`, Part 1)*

### 4.1 Key words

* **Classification** – *what* is in the image? ("cat")
* **Detection** – *what* and *where*, for every object → a **bounding box** (x, y, width, height) + class label.
* **Segmentation** – the exact pixels of each object.
* **Confidence score** – how sure the model is (0 to 1).
* **True positive (TP)** – a correct detection. **False positive (FP)** – a wrong detection (a "face" on a curtain). **False negative (FN)** – a missed object.
* **Precision = TP / (TP + FP)** – of everything detected, how much was right?
* **Recall = TP / (TP + FN)** – of all real objects, how many were found?
* **IoU (Intersection over Union)** = overlap area of two boxes ÷ their combined area. 1 = same box, 0 = no overlap.
* **mAP (mean Average Precision)** – the standard accuracy score for detectors, averaged over classes.

### 4.2 Viola–Jones (Haar cascade)

By Paul Viola and Michael Jones (2001) – the first face detector fast enough for real time; it was used in digital cameras.
Four key ideas:

**1. Haar-like features** – simple black-and-white rectangle patterns.
Value of a feature = (sum of pixels under the white part) − (sum under the black part).
Types: two-rectangle (edge), three-rectangle (line), four-rectangle (diagonal).
Example: the **eye region is darker than the cheeks** below it; the **nose bridge is brighter than the eyes** on both sides.

**2. Integral image** – a trick to compute rectangle sums super fast.
Each position of the integral image stores the **sum of all pixels above and to the left** of it.
Then the sum of **any** rectangle needs only **4 lookups**: `D − B − C + A` (the four corners), no matter how big the rectangle is.

```
image        integral image
1 2 3        1  3  6
4 5 6        5 12 21
7 8 9       12 27 45      sum of bottom-right 2×2 block (5+6+8+9 = 28) = 45 − 6 − 12 + 1 = 28
```

**3. AdaBoost (choosing the best features)** – a 24 × 24 window has more than 160,000 possible Haar features – far too many. AdaBoost picks the few most useful ones and combines many **weak classifiers** (each is one feature + a threshold, only a bit better than guessing) into one **strong classifier**.

**4. Cascade** – the classifier is a chain of stages. Early stages are tiny (a couple of features) and quickly **reject** windows that are obviously not a face. Only promising windows go on to the later, more detailed stages.
Like airport security: a quick check for everyone, a detailed check only for suspicious bags. Since most of an image is background, this is very fast.

**Our code:**

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

* `cv2.data.haarcascades` – the folder where OpenCV keeps its pretrained XML files.
* **scaleFactor = 1.1** – the image is shrunk by 10% each round so faces of every size fit the fixed detector window. Smaller value (1.05) = more accurate but slower.
* **minNeighbors = 5** – a face must be detected by at least 5 overlapping windows. Higher = fewer false positives, but may miss faces.
* **minSize = (30, 30)** – ignore detections smaller than 30 × 30 pixels.
* Output: a list of `(x, y, w, h)` boxes. **No confidence score** – Haar only says yes/no.
* Other pretrained cascades in OpenCV: eyes, smile, profile face, full body, upper body, cat face, number plates (`haarcascade_eye.xml`, `haarcascade_fullbody.xml`, …).

**Limitations:** only upright, front-facing faces; sensitive to lighting; false positives on textures; one object type per cascade file; no confidence score.

### 4.3 YOLO (You Only Look Once)

By Joseph Redmon et al. (2016). Older deep-learning detectors (R-CNN family) first proposed ~2,000 regions and classified each one – slow. YOLO looks at the image **once**, with **one pass** through a neural network, and predicts all boxes at the same time → real-time speed.

**CNN (Convolutional Neural Network)** – a neural network that **learns its own kernels (filters) from data** instead of using hand-made features like Haar or HOG. Early layers learn edges, middle layers learn parts (eyes, wheels), and the last layers recognise whole objects.

**How YOLO predicts (simple version):**
1. The image is resized to 640 × 640.
2. The network divides it into a **grid**. Each grid cell predicts objects whose centre falls inside it: box position and size (x, y, w, h), and the probability of each of the 80 classes.
3. YOLOv8 predicts on 3 grid sizes – coarse for big objects, fine for small objects – and is **anchor-free** (it predicts boxes directly instead of adjusting pre-defined template boxes).
4. **Confidence threshold** – drop boxes with confidence below `conf=0.25`.
5. **NMS (Non-Maximum Suppression)** – the network often predicts several overlapping boxes for the same object. NMS keeps the one with the highest confidence and removes the others that overlap it too much (high IoU).

**Pretrained on COCO** ("Common Objects in Context") – about 118,000 training images, **80 classes** (person, bicycle, car, dog, chair, bottle, cell phone, laptop, …).

**Model sizes:** n (nano), s, m, l, x – bigger = more accurate but slower. `yolov8n.pt` is the smallest (~3 million parameters, ~6 MB).

**Our code:**

```python
model = YOLO("yolov8n.pt")                          # load pretrained model (downloads once)
result = model(img, conf=0.25, verbose=False)[0]    # run detection on one image
for box in result.boxes:
    name = model.names[int(box.cls[0])]             # class number → name, e.g. 0 → "person"
    conf = float(box.conf[0])                       # confidence 0..1
annotated = result.plot()                           # image with boxes, labels and scores drawn
```

* We do one **warm-up** run first because the very first call is slower (loading the model into memory), which would make the timing unfair.
* YOLO can be **fine-tuned** on your own classes with `model.train(data="my_data.yaml")` (transfer learning).

**Limitations:** very small or very dark objects; unusual angles; crowded overlapping objects; it can only name the 80 classes it was trained on – an unknown object may get the closest wrong label (e.g. a TV remote called "cell phone"); needs a GPU for the best speed; training needs lots of labelled data.

### 4.4 Viola–Jones vs YOLO

| | Viola–Jones (Haar) | YOLO (v8n) |
|---|---|---|
| Year / type | 2001, classic machine learning (hand-made Haar features + AdaBoost) | 2016+ (v8 in 2023), deep learning (CNN) |
| Classes | one per cascade (here: frontal face) | 80 COCO classes at once |
| Output | box only | box + label + confidence |
| Accuracy | lower; many false positives/misses | much higher |
| Rotation / side faces | misses them | still finds the person |
| Speed on CPU | light, no GPU needed; slows down on big images and small `scaleFactor` | nano model is quick on a modern CPU; very fast on a GPU |
| Size | ~1 MB XML | ~6 MB weights |
| Training | needs thousands of positive/negative images, slow to train | needs a big labelled dataset + GPU |

**About speed – a surprise worth mentioning in the viva:** textbooks say Viola–Jones is "very fast", and it was in 2001 on small 384 × 288 images. But `detectMultiScale` checks the image at many sizes (with `scaleFactor=1.1`, an 800-pixel image is scanned at about 35 scales), so on large images it slows down. In our test on an 800-pixel photo, the Haar cascade took roughly 300–600 ms while YOLOv8-nano took about 100–200 ms on the same laptop CPU. Modern CNNs like YOLOv8n are heavily optimised, so "classic = faster" is not always true. Use your own timing table as evidence.

### 4.5 Failure cases — and how to explain them

| Case | What usually happens | Why |
|---|---|---|
| Rotated 90° | Haar **misses** the face; YOLO still detects a person (maybe lower confidence) | Haar was trained on upright faces; its rectangle features are aligned with the image axes |
| Very dark | both lose detections or confidence drops | low contrast → Haar's light/dark differences become tiny; YOLO sees less detail |
| Very small | both may miss the face/person | smaller than Haar's `minSize` / training window; YOLO loses detail when objects are only a few pixels |
| Patterned background | Haar finds a "face" that is not there (false positive) | by chance the pattern has dark-light areas like eyes and nose |
| Unusual object | YOLO gives a wrong class | the object is not one of the 80 COCO classes, so YOLO picks the most similar one |

In the report, write what **you** saw, e.g. *"Haar detected 0 faces in the rotated image while YOLO still detected 'person' with 0.81 confidence."*

### 4.6 Interview questions — object recognition

1. **Classification vs detection?** – Classification says *what*; detection says *what and where* (with boxes).
2. **What are Haar-like features?** – Black-and-white rectangle patterns; value = sum under white − sum under black.
3. **What is an integral image and why is it useful?** – Each position stores the sum of everything above-left; any rectangle sum needs only 4 lookups → very fast feature computation.
4. **What is AdaBoost's job in Viola–Jones?** – Choose the few best features out of 160,000+ and combine weak classifiers into a strong one.
5. **What is the cascade?** – A chain of increasingly complex stages; easy negatives are rejected early, which makes it fast.
6. **What do `scaleFactor` and `minNeighbors` do?** – `scaleFactor`: how much the image shrinks each round to find different face sizes. `minNeighbors`: how many overlapping detections are needed to accept a face (higher = fewer false positives).
7. **Why doesn't Haar give a confidence score?** – The cascade only says pass/fail at each stage; `detectMultiScale` returns boxes only.
8. **What does YOLO stand for and why is it fast?** – You Only Look Once; one forward pass of a CNN predicts all boxes at once.
9. **What is a confidence score?** – How sure the model is that the box contains that class.
10. **What is NMS?** – Non-Maximum Suppression: among overlapping boxes for the same object, keep the highest-confidence one and remove the rest.
11. **What is IoU?** – Intersection over Union: overlap ÷ union of two boxes; used in NMS and to judge whether a detection is correct.
12. **What is COCO?** – A large dataset with 80 everyday object classes; YOLO's pretrained weights come from it.
13. **What does "pretrained" mean?** – The model was already trained on a big dataset; we just use the learned weights.
14. **Precision vs recall?** – Precision: how many detections were correct. Recall: how many real objects were found.
15. **What is a false positive / false negative?** – FP: detecting something that is not there. FN: missing a real object.
16. **When would you still use Haar today?** – On very weak hardware without a GPU, when you only need simple frontal face detection.
17. **What do n/s/m/l/x mean in YOLO?** – Model sizes from nano to extra-large: bigger = more accurate but slower.

---

## 5. Unit 5B — Motion estimation

*(Notebook: `Unit5_Recognition_Motion.ipynb`, Part 2)*

### 5.1 Video basics

* A video is a sequence of images called **frames**, shown quickly – e.g. **30 frames per second (fps)**.
* Duration = number of frames ÷ fps (e.g. 600 frames ÷ 30 fps = 20 s).
* `cv2.VideoCapture(path)` opens the video; `cap.read()` returns `(ok, frame)` – `ok` is `False` when the video ends.
* **Why keep the camera still?** Frame differencing and background subtraction assume a **fixed camera**. If the camera moves, every pixel changes and everything looks like it is moving.

### 5.2 Frame differencing

**Idea:** if a pixel changed between two frames, something moved there.

```python
diff = cv2.absdiff(g1, g2)                                    # |frame2 − frame1| for every pixel
_, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)     # changed by more than 25 → moving
mask = cv2.dilate(mask, kernel, iterations=2)                 # join broken pieces
```

* We convert to gray and **blur** first so tiny noise and video compression artefacts are ignored.
* `absdiff` = absolute difference (always positive).
* The threshold 25 separates real motion from small noise.

**Problems:**
1. **Hollow objects** – inside a plain-coloured shirt, consecutive frames look the same, so only the **edges** of the moving person show up.
2. **Stops when motion stops** – a person standing still disappears.
3. **Ghosting** – a fast object appears twice (old and new position).
4. **Camera shake / lighting flicker** – the whole image looks like it is moving.

Variants: *three-frame differencing* (combine two differences to reduce ghosting); difference with a fixed empty-background photo.
**Strength:** the simplest and fastest method.

### 5.3 GMM background subtraction (MOG2)

**Background subtraction:** foreground = current frame compared with a **model of the background**.

Why not just take one photo of the empty scene? Because real backgrounds change: lighting changes, trees sway, screens flicker, a chair is moved.

**Gaussian Mixture Model (GMM):** for every single pixel we keep a few **bell curves (Gaussians)** that describe the colours that pixel usually has. Each curve has:
* a **mean** (the typical colour),
* a **variance** (how much it varies),
* a **weight** (how often this colour is seen).

Example: a pixel on a wall → one narrow curve "light grey". A pixel on a waving leaf → two curves "green" and "brown". Curves with big weight and small variance = **background**.

When a new frame comes:
* if the pixel's colour fits one of its background curves → **background** (black),
* if not → **foreground** (white) – something new is there,
* then the curves are **updated** with the new value, so the model slowly adapts (learns).

Parameters in our code:
* `history=200` – how many recent frames the model "remembers" (learning rate ≈ 1/history).
* `varThreshold=16` – how far a colour must be from the background curves to count as foreground (squared distance 16 → about √16 = **4 standard deviations**).
* `detectShadows=True` – a pixel that looks like a **darker version** of the background colour is marked as **shadow = 127 (grey)**. We remove shadows with `threshold(raw, 200, …)`.
* `getBackgroundImage()` – shows what the model thinks the empty scene looks like.

MOG2 (by Zivkovic, 2004) improves on the older MOG: it **automatically chooses how many Gaussians** each pixel needs and **detects shadows**.

**Behaviour to know:**
* The first few frames are noisy – the model is still learning.
* If a person stands still for a long time, they slowly **fade into the background**.
* Another OpenCV option: `cv2.createBackgroundSubtractorKNN`.

### 5.4 Optical flow and Lucas–Kanade

**Optical flow** = the apparent motion of brightness patterns between two frames. For each point it gives a **motion vector (u, v)** = how many pixels it moved in x and in y.

**Three assumptions:**
1. **Brightness constancy** – a point keeps the same brightness when it moves.
2. **Small motion** – points move only a little between frames.
3. **Spatial coherence** – neighbouring pixels move together.

From assumption 1 we get the **optical flow equation**:

```
Ix · u  +  Iy · v  +  It  =  0
```

Ix, Iy = image gradients in x and y; It = change over time; u, v = the unknown motion.

**The aperture problem:** this is **one equation with two unknowns** – it cannot be solved from a single pixel. (Like looking at a moving line through a small hole: you can only tell the motion perpendicular to the line – the barber-pole illusion.)

**Lucas–Kanade's solution (1981):** use assumption 3 – assume all pixels in a small window (we use **15 × 15 = 225 pixels**) move the same way. Now there are 225 equations for 2 unknowns → solve with **least squares** (the best-fitting answer).
This only works well if the window contains gradients in **two different directions** – i.e. a **corner**. On a flat area or along a straight edge, it fails.

**That is why we track corners** – `cv2.goodFeaturesToTrack` uses the **Shi–Tomasi** corner detector:
* `maxCorners=300` – at most 300 points,
* `qualityLevel=0.01` – keep corners at least 1% as strong as the strongest one,
* `minDistance=7` – points at least 7 pixels apart.

**Pyramidal LK** (`calcOpticalFlowPyrLK`, `maxLevel=2`): if an object moves fast, the "small motion" assumption breaks. So we make an **image pyramid** – the image at full, ½ and ¼ size. At ¼ size, a 20-pixel motion becomes only 5 pixels (small again). LK estimates motion on the small image first and refines it on the bigger ones.

Outputs: new point positions, and `status` (1 = point found again, 0 = lost).
In our arrow picture, arrows are drawn **5× longer** than the real motion, only so they are visible. We also draw the **paths** of points tracked over 20 frames.

**Sparse vs dense optical flow:**
* **Sparse** (Lucas–Kanade) – only for selected points; fast.
* **Dense** (e.g. Farnebäck, `cv2.calcOpticalFlowFarneback`) – a vector for **every** pixel; slower; often shown as a colour image (colour = direction, brightness = speed).

**Uses:** object tracking, video stabilisation, measuring speed and direction, action recognition, self-driving cars.
**Limitations:** lighting changes, very fast motion, points hidden behind other objects (occlusion), plain areas with no corners, drift over long tracking.

### 5.5 Comparing the three methods

| | Frame differencing | GMM (MOG2) | Lucas–Kanade |
|---|---|---|---|
| Question it answers | "did this pixel change?" | "is this pixel different from the usual background?" | "where did this point move, and how far?" |
| Output | binary mask | binary mask (+ grey shadows) | motion vectors (direction + speed) |
| Needs learning time? | no | yes | no |
| Object shape | outline only (hollow inside) | full silhouette | only scattered points |
| When the object stops | vanishes at once | stays for a while, then fades | vectors ≈ 0 |
| Camera shake | everything looks like motion | many false pixels until it re-learns | every point moves (can measure the camera motion) |
| Speed | fastest | fast | slower |
| Best for | simple, quick motion alarm | separating people/cars from a fixed background (CCTV) | measuring direction and speed, tracking |

The notebook prints a table with the **average time per frame** and the **average % of moving pixels** (or average motion in pixels), and plots both over time. Usually **frame differencing marks fewer pixels** than GMM (because it only sees edges), and GMM gives fuller shapes.

In the "foreground size over time" chart, the **GMM line starts near 0 and rises during the first 1–2 seconds** – that is the **learning phase**, while MOG2 is still building its background model. Frame differencing has no learning phase, so its line is "ready" from the first frame. This is a good point to mention in the video.

### 5.6 Interview questions — motion estimation

1. **What is frame differencing?** – Subtracting consecutive frames and thresholding the absolute difference to find pixels that changed.
2. **Why are frame-differencing masks hollow?** – Inside a uniformly coloured moving object, consecutive frames look the same, so only the edges change.
3. **What is background subtraction?** – Comparing each frame with a learned model of the background; differences are foreground.
4. **What is a Gaussian Mixture Model?** – A combination of several bell curves; here, for each pixel, describing the colours it usually has.
5. **Why use a mixture instead of one Gaussian?** – Some background pixels have more than one normal colour (waving trees, flickering screens).
6. **What do `history` and `varThreshold` mean in MOG2?** – History: how many past frames the model remembers. varThreshold: how far a pixel must be from the background model to count as foreground.
7. **How does MOG2 handle shadows?** – A pixel that is a darker version of the background colour is labelled as shadow (value 127), and we remove it with a threshold.
8. **What happens if a person stops moving?** – Frame differencing loses them immediately; MOG2 slowly absorbs them into the background.
9. **What is optical flow?** – The apparent motion of pixels between frames, given as (u, v) vectors.
10. **What are the assumptions of Lucas–Kanade?** – Brightness constancy, small motion, and neighbouring pixels moving together.
11. **What is the aperture problem?** – One pixel gives one equation for two unknowns, so motion along an edge cannot be determined; you need a corner or a window of pixels.
12. **Why do we track corners?** – Corners have gradients in two directions, so their motion can be solved uniquely.
13. **Why use an image pyramid?** – To handle large motions: on a smaller image, a big motion becomes small enough for LK.
14. **Sparse vs dense optical flow?** – Sparse: selected points (Lucas–Kanade). Dense: every pixel (Farnebäck).
15. **Which method would you use for a CCTV camera?** – GMM background subtraction – it gives full silhouettes and adapts to slow lighting changes.
16. **Which method gives the direction of motion?** – Only optical flow; the other two only say *where* something moved.

---

## 6. One-page cheat sheet

| Technique | One-line idea | OpenCV / library call | Key parameter(s) |
|---|---|---|---|
| Gaussian noise | add bell-curve random values to every pixel | `np.random.normal` | σ = 25 |
| Salt-and-pepper | random pixels → 0 or 255 | `np.random.rand` | 10% density |
| Gaussian filter | weighted average of neighbours | `cv2.GaussianBlur` | kernel 5 × 5, sigma |
| Median filter | middle value of sorted neighbours | `cv2.medianBlur` | kernel 5 |
| PSNR | pixel error in dB | `skimage.metrics.peak_signal_noise_ratio` | data_range = 255 |
| SSIM | structural similarity 0–1 | `skimage.metrics.structural_similarity` | channel_axis = 2 |
| Otsu threshold | automatic best split of dark/bright | `cv2.threshold(…, THRESH_OTSU)` | – |
| Erosion / dilation | shrink / grow white regions | `cv2.erode`, `cv2.dilate` | kernel 5 × 5 |
| Opening / closing | remove specks / fill holes | `cv2.morphologyEx` | `MORPH_OPEN`, `MORPH_CLOSE` |
| Canny | thin, connected edges | `cv2.Canny` | low, high thresholds |
| Watershed | flood from markers, dams = boundaries | `cv2.distanceTransform`, `cv2.watershed` | FG_RATIO = 0.5 |
| HOG | histograms of edge directions in cells | `skimage.feature.hog` | 9 bins, 16 × 16 cells, 2 × 2 blocks |
| SIFT | scale/rotation-invariant keypoints, 128-D descriptor | `cv2.SIFT_create().detectAndCompute` | – |
| Matching | nearest descriptors + ratio test + RANSAC | `cv2.BFMatcher.knnMatch`, `cv2.findHomography` | ratio 0.75, RANSAC 5 px |
| Viola–Jones | Haar features + integral image + AdaBoost + cascade | `cv2.CascadeClassifier.detectMultiScale` | scaleFactor 1.1, minNeighbors 5 |
| YOLO | one CNN pass predicts boxes + classes | `ultralytics.YOLO("yolov8n.pt")` | conf = 0.25 |
| Frame differencing | pixels that changed since last frame | `cv2.absdiff` + `cv2.threshold` | threshold 25 |
| GMM (MOG2) | per-pixel mixture of Gaussians background model | `cv2.createBackgroundSubtractorMOG2` | history 200, varThreshold 16 |
| Lucas–Kanade | track corners between frames | `cv2.goodFeaturesToTrack`, `cv2.calcOpticalFlowPyrLK` | window 15 × 15, 2 pyramid levels |

---

## 7. How the project is organised

```
CompVision/
├── data/                                  ← your own input files (see data/README.md)
│   ├── photo1.jpg                         main photo
│   ├── photo2.jpg                         same scene, different angle (Unit 4)
│   ├── objects.jpg                        optional: touching coins/objects (watershed)
│   └── video.mp4                          15–30 s video with motion (Unit 5)
├── outputs/                               ← created automatically when you run the notebooks
│   ├── unit3/  unit4/  unit5/
├── Unit3_Restoration_Segmentation.ipynb
├── Unit4_Feature_Extraction.ipynb
├── Unit5_Recognition_Motion.ipynb
├── notes.md                               ← this file
├── REPORT.md                              ← technical report template
├── README.md                              ← setup and run steps
└── requirements.txt
```

Each notebook follows the same pattern, so the code is easy to read:
1. imports and three small helper functions (`load_image`/`resize`, `show`, `save`),
2. one section per step of the task, each with a short explanation cell above the code,
3. tables printed in Markdown format (copy-paste into the report),
4. an observation / comparison cell at the end.

---

## 8. Script for the 5–8 minute explanation video

You must speak in your own words and show your face or voice (a silent screen recording is not accepted). A suggested plan:

| Time | What to show | What to say |
|---|---|---|
| 0:00 – 0:40 | yourself / title slide | your name, the assignment (Units 3, 4, 5), the goal: restoration, segmentation, features, recognition, motion |
| 0:40 – 1:20 | `data/` folder: photo1, photo2, video | where and how you captured them, what is in them, why they are suitable (details, a moving person, stable camera) |
| 1:20 – 3:00 | Unit 3 notebook | noise types → filters → PSNR/SSIM table ("median wins for salt-and-pepper because…") → threshold & morphology → Canny settings → watershed count |
| 3:00 – 4:20 | Unit 4 notebook | HOG picture and vector length → SIFT keypoints (circle = scale, line = orientation) → matches → rotation/resize table ("SIFT is rotation- and scale-invariant because…") |
| 4:20 – 5:40 | Unit 5 Part 1 | Haar vs YOLO results, timing table, a failure case you found and **why** it happened |
| 5:40 – 7:00 | Unit 5 Part 2 + `motion_comparison.mp4` | frame difference (hollow), GMM (full silhouette, shadows), optical-flow arrows and paths, comparison table |
| 7:00 – 7:40 | summary | best technique for each job, what you learned, Git link |

Tips: run all notebooks before recording so outputs are visible; zoom the editor so code is readable; scroll slowly; explain **why** a result happened, not just what it is.

---

## 9. Glossary

| Word | Simple meaning |
|---|---|
| **AdaBoost** | combines many weak, simple classifiers into one strong classifier |
| **Aperture problem** | from a small area you cannot tell motion along an edge; you need corners |
| **Binary image** | an image with only black (0) and white (255) |
| **Bounding box** | rectangle around a detected object (x, y, width, height) |
| **Cascade** | a chain of classifier stages that rejects easy negatives early |
| **Channel** | one colour layer of an image (B, G or R) |
| **CNN** | Convolutional Neural Network – a neural network that learns its own image filters |
| **COCO** | big dataset with 80 everyday object classes |
| **Confidence score** | how sure a detector is about a detection (0–1) |
| **Convolution** | sliding a kernel over an image and computing a weighted sum at each position |
| **Descriptor** | numbers that describe the area around a keypoint |
| **Distance transform** | for every white pixel, the distance to the nearest black pixel |
| **DoG** | Difference of Gaussians – subtract two blurred images to find blobs (SIFT) |
| **Foreground** | the moving / interesting part of a scene (opposite of background) |
| **fps** | frames per second |
| **Gaussian** | the bell-shaped curve (normal distribution) |
| **Gradient** | how fast and in which direction brightness changes |
| **Haar feature** | difference between pixel sums of black and white rectangles |
| **Histogram** | counts of how often each value (brightness or direction) appears |
| **Homography** | 3 × 3 matrix mapping one view of a plane to another |
| **Hysteresis** | Canny's rule: keep weak edges only if they connect to strong edges |
| **Inlier / outlier** | a match that agrees / disagrees with the best transformation |
| **Integral image** | table of cumulative sums so any rectangle sum takes 4 lookups |
| **IoU** | Intersection over Union – overlap of two boxes |
| **Kernel** | small grid of numbers used as a sliding window (also "filter" or "structuring element") |
| **Keypoint** | an interesting, repeatable point (corner/blob) in an image |
| **Mask** | binary image marking which pixels belong to something |
| **MSE** | Mean Squared Error – average squared pixel difference |
| **NMS** | Non-Maximum Suppression – keep the best of overlapping boxes / edge pixels |
| **Noise** | random, unwanted changes in pixel values |
| **Octave** | one image size level in SIFT's pyramid (full, ½, ¼ …) |
| **Optical flow** | apparent motion of pixels between frames |
| **Otsu** | automatic threshold that best separates dark and bright pixels |
| **Pixel** | the smallest square of a digital image |
| **Pretrained** | already trained on a big dataset, ready to use |
| **PSNR** | Peak Signal-to-Noise Ratio – quality in dB, higher is better |
| **Pyramid** | the same image at several smaller sizes |
| **RANSAC** | finds the model most matches agree on by testing random samples |
| **Ratio test** | keep a match only if it is clearly better than the second-best |
| **Segmentation** | splitting an image into meaningful regions |
| **SSIM** | Structural Similarity – quality from 0 to 1, closer to human perception |
| **Structuring element** | the kernel shape used in morphology |
| **Threshold** | cut-off value that decides black or white |
| **uint8** | whole numbers from 0 to 255 – the normal image data type |
