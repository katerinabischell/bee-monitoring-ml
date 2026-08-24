# Methods

## 1. Study system

Two federally endangered plant species were monitored at the North Campus Open Space
(NCOS) restoration site in Goleta, California:

- **Ventura marsh milk-vetch** — *Astragalus pycnostachyus* var. *lanosissimus*
- **Salt marsh bird's-beak** — *Chloropyron maritimum* ssp. *maritimum*

Both are pollinated primarily by bumblebees (*Bombus* spp., including
*B. vosnesenskii*). Their visitation ecology is poorly documented, in large part
because direct human observation is labour-intensive, weather-dependent, and does not
scale to the number of observation hours needed to characterise visitation rates.

Site coordinates are deliberately not published in this repository. NCOS is a
publicly accessible restoration site, but fine-grained locality data for listed
species is withheld as a matter of practice.

## 2. Field data collection

Collection ran for **nine weeks, June–September 2025**, across **four sites**
(quadrants) within the restoration area. Sites were marked with neon orange or yellow
yarn to define a consistent camera frame between sessions.

- **Schedule.** Three days per week, three shifts per day: morning (08:00–12:00),
  mid-day (12:00–15:00), and afternoon (15:00–18:00), yielding roughly three hours of
  footage per collection day.
- **Site order** was randomised (1–4) per session using a random number generator, so
  that time of day was not confounded with site.
- **Cameras.** Two Panasonic Lumix cameras per shift, positioned a short distance from
  the marked quadrant, zoomed to the quadrant edges and angled downward to capture
  pollinators moving above the plant canopy. A separate long-duration camera was set
  up in the morning and left in place until afternoon for extended coverage.
- **Total footage.** Approximately 200 hours across the season.

### Environmental and ground-truth log

Every recording session was logged by hand. The log is included at
[`field_log.csv`](field_log.csv) (86 rows) and records, per session: date, week, site
number, camera number, start and end time, weather and conditions, wind speed and
direction (pouch anemometer), observed pollinator activity on and around the site,
video duration, shift type, and free-text notes.

The notes column carries the operational detail that makes the record usable —
equipment failures, memory cards filling mid-session, changes in cloud cover partway
through a recording, and incidental observations of non-target taxa such as *Bembix*.
This log is the human ground truth against which automated detections are compared,
and it covers both target species, since collection for the two ran in parallel.

## 3. Frame extraction and annotation

The detection work in this repository draws on a single recording,
**`P1000471.MP4`** (Week 6, Day 1, Site 1, morning shift; 21 minutes). See
[`limitations.md`](limitations.md) for what that implies.

**Extraction.** Frames were sampled from the video at 1–2 second intervals using
OpenCV (`src/extract_frames.py`).

**Placeholder scaffolding.** The observation log records, for each timestamp, whether
a bee was present and which cell of a 3×3 quadrant grid it occupied. Extraction seeds
one placeholder box per recorded bee at the centre of the corresponding grid cell.
These placeholders identify *which frames need annotation*; they are not training
data, and models are never trained on them directly.

**Manual annotation.** Each flagged frame was opened in a Matplotlib annotation GUI
(`src/annotate.py`) and the placeholder replaced with a tightly drawn box around the
bee. The tool records progress so that the pass can be resumed across sittings. The
annotated pool is **126 frames containing 157 boxes**. Recorded annotation effort was
approximately seven hours across two passes, including frames later excluded from the
final dataset.

**Visual QA.** Every saved label was rendered back onto its frame
(`src/visualize_annotations.py`) and reviewed by eye. Bees occupy roughly 1–2% of
frame width, so drifted or leftover placeholder boxes are not reliably visible in
summary statistics.

### Split construction

Because all frames come from one continuous recording sampled every 1–2 seconds, a
shuffled split places near-duplicate frames on both sides and yields an optimistic
estimate. The evaluation split is therefore **temporal**: frames are ordered by
position in the source video and cut into contiguous blocks, with a buffer discarded
between blocks so that no evaluation frame lies within ~30 seconds of a training
frame.

| Split | Time range in source video | Frames |
|---|---|---|
| Train | 00:26 – 16:12 | 79 |
| *(buffer, 35 s discarded)* | | |
| Validation | 16:47 – 18:31 | 23 |
| *(buffer, 41 s discarded)* | | |
| Test | 19:12 – 21:02 | 11 |

113 of the 126 annotated frames are retained; 13 fall inside the buffers and are
discarded. The exact frame lists are in `configs/split_temporal.json`.

## 4. Model training

YOLOv8-small (Ultralytics), initialised from COCO-pretrained weights and fine-tuned
on the annotated frames. Single class, `bee`.

Key hyperparameters (full set in `configs/train_v2.yaml`): 150 epochs, patience 30,
batch 8, image size 640, `rect=False`, `lr0` 0.01 with `lrf` 0.1, SGD momentum 0.937,
weight decay 0.0005, 3 warmup epochs, mosaic augmentation closed for the final 10
epochs, seed 0, deterministic.

Six training runs were carried out over the course of the project, including a
yolov8n baseline and several fine-tunes initialised from earlier checkpoints. All six
are reported in the results table; the fine-tunes did not improve on the yolov8s run.

Training environment: Python 3.11.0, PyTorch 2.7.1, Ultralytics 8.3.170, CPU
(Apple silicon). Pinned versions are in `requirements.txt`.

## 5. Detection, verification, and tracking

**Inference** over field video (`src/evaluate.py`) samples frames at fixed intervals
and records per-frame bee counts with confidence statistics, keyed by week, day, site,
and shift so that detections can be related back to the field design.

**Verification** (`src/verify_detections.py`) filters detections by confidence and
renders a sample for human review. This analysis supports the **0.4 confidence
threshold** used elsewhere: of 89 raw detections, 57 survive a 0.3 threshold and 16
exceed 0.5, with mean confidence 0.387.

**Compilation** (`src/compile_detections.py`) writes annotated frames plus
`detection_metadata.csv`, one row per detected box with timestamp, confidence, and
pixel coordinates.

**Tracking.** `src/track_hydat.py` implements a hybrid detection-and-tracking pass
adapted from the HyDaT approach of Ratnayake et al. (2021): the YOLOv8 detector is
combined with a
background-subtraction fallback, and detections are linked across frames by Hungarian
assignment so that a bee is followed through the frame rather than counted
independently in each one. Short detection gaps are bridged so a brief miss does not
split one visit into two. `src/track_hydat_lowres.py` adds a downscaled detection pass
intended to catch fast-moving bees lost to motion blur.

Tracking is the stage that converts per-frame detections into **visitation events**.
Visitation-event statistics are not yet reported; the tracker runs, but its output has
not been validated against the hand-built log.

## 6. Planned work

- Annotate held-out video from other days, sites, and lighting conditions, and
  measure generalization properly.
- Implement and evaluate synthetic annotation — compositing bumblebee cutouts onto
  unoccupied vegetation stills with automatically derived bounding boxes — and compare
  it against manual annotation on equal footing. Not yet implemented; see
  [`limitations.md`](limitations.md).
- Validate tracker output against the field log and report visitation rates.

## References

Ratnayake, M.N., Dyer, A.G., & Dorin, A. (2021). Tracking individual honeybees among
wildflower clusters with computer vision-facilitated pollinator monitoring. *PLOS ONE*,
16(2), e0239504. https://doi.org/10.1371/journal.pone.0239504
