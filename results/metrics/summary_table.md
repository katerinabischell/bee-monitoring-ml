# Results summary

Generated from the per-epoch training logs in this directory and, for the
temporal-holdout run, from a fresh evaluation on a disjoint split.

## Headline

**Temporal holdout (disjoint split, no near-duplicate overlap):**

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation | **0.632** | **0.236** | 0.735 | 0.684 |
| Test | 0.439 | 0.175 | 0.927 | 0.364 |

yolov8s with the `v2` hyperparameters, trained from COCO weights on 79 frames
(00:26-16:12 of the source video), validated on 23 frames (16:47-18:31) and tested
on 11 frames (19:12-21:02), with 35 s and 41 s buffers between blocks. Early-stopped
at epoch 72 of 150 (patience 30); best epoch 42.
Evaluated under the Ultralytics default validation protocol, the same protocol used
by every training log below.

**This is the figure to cite.** Two caveats, both in
[`../../docs/limitations.md`](../../docs/limitations.md):

- The validation set is 23 frames / 38 boxes. The estimate is unbiased but
  imprecise, and should be read as an order-of-magnitude check rather than a
  precise performance figure.
- **This is not a like-for-like correction of the 0.745 below.** The temporal
  holdout trains on 79 frames where the original run had 122, a 35% reduction.
  Removal of the leakage and the smaller training set both push the score down,
  and this experiment cannot separate the two effects.

## All training runs (training-fit, non-disjoint split)

> These six runs were all trained and validated before the split overlap described
> in [`../../docs/limitations.md`](../../docs/limitations.md) was identified. In the
> split then in use, 53 of 57 validation frames and 34 of 35 test frames were also
> in training. **The numbers below are training-fit diagnostics, not held-out
> performance,** and are retained for provenance and for comparison between runs.

| Run | Backbone | Initialised from | Epochs (ran/planned) | Best epoch | mAP50 | mAP50-95 | P | R | Final-epoch mAP50 / mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|
| **v1** | yolov8n | COCO | 50/50 | 41 | 0.454 | 0.169 | 0.728 | 0.406 | 0.455 / 0.162 |
| **v2** ⭐ | yolov8s | COCO | 150/150 | 131 | 0.746 | 0.312 | 0.875 | 0.648 | 0.631 / 0.203 |
| **v23** | yolov8n | v1 checkpoint | 72/200 | 42 | 0.479 | 0.167 | 0.717 | 0.459 | 0.412 / 0.129 |
| **v232** | yolov8n | v23 checkpoint | 200/200 | 132 | 0.635 | 0.262 | 0.766 | 0.467 | 0.601 / 0.195 |
| **v24** | yolov8s | v23 checkpoint | 8/400 | 2 | 0.475 | 0.171 | 0.719 | 0.430 | 0.391 / 0.137 |
| **v25** | yolov8s | v23 checkpoint | 2/200 | 2 | 0.477 | 0.163 | 0.689 | 0.412 | 0.477 / 0.163 |

⭐ `v2` is the run referred to elsewhere as the published model. Its best epoch
(131 of 150) records mAP50 0.745, mAP50-95 0.312 — the figure previously reported
for this work. The checkpoint `models/ventura_bee_detection_v2_best.pt` is that
epoch; the values are embedded in the checkpoint's own `train_metrics`.

Runs `v23` through `v25` were fine-tunes initialised from earlier checkpoints at a
reduced learning rate. None improved on `v2`. `v24` and `v25` were stopped early
(8 and 2 epochs of a planned 400 and 200). A seventh run, `v22`, produced a
configuration file but no metrics and is not listed.

## Detection output on field video

From `detection_summary_stats.txt` and `verification_summary.txt`, using the
`v2` checkpoint at a 0.4 confidence threshold across 6 videos:

- 67 frames containing at least one detection; 89 individual bee detections
- Mean confidence 0.387, range 0.251-0.766
- Mean 1.33 bees per detection frame; maximum 4 in a single frame
- Of 89 raw detections, 57 exceed confidence 0.3 and 16 exceed 0.5

These are descriptive counts of detector output, not accuracy measurements.
