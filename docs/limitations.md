# Limitations

This section records the constraints on the evaluation reported in this repository.
They bear directly on how the reported detection metrics should be interpreted.

## 1. The originally reported metric is a training-fit score

An earlier version of this work reported the detection model's performance as
**mAP50 0.745, mAP50-95 0.312**, taken from epoch 131 of the 150-epoch
`ventura_bee_detection_v2` run. That figure is reproducible and is embedded in the
checkpoint itself, but it does not describe held-out performance.

The dataset split in use when that run was trained was not disjoint. Of the 57
frames assigned to validation, **53 were also present in the training set**; of the
35 frames assigned to test, **34 were also in the training set**. The model was
therefore validated largely against frames it had been trained on.

This was confirmed by re-running validation on the archived checkpoint against the
split as it stood at training time. The recorded training-log recall and the
recomputed recall agree to five decimal places (**0.64789** in both cases), which
identifies the split unambiguously:

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Training log, epoch 131 | 0.74550 | 0.31176 | 0.87537 | 0.64789 |
| Recomputed on same split | 0.74281 | 0.29943 | 0.88497 | 0.64789 |

The 0.745 figure is retained in the results table for provenance, labelled as a
training-fit diagnostic. It is not a measure of generalization and should not be
cited as one.

## 2. All annotated data derives from a single video

Every annotated frame in this repository comes from **one 21-minute recording
(`P1000471.MP4`), from one camera, at one site, on one day**, sampled at 1–2 second
intervals. The consequences are structural rather than incidental:

- Consecutive frames are near-duplicates. Across the annotated pool the median gap
  between a frame and its nearest neighbour is 5 seconds.
- A randomly shuffled split therefore cannot separate training from evaluation data
  in any meaningful sense, regardless of the split ratio, because visually near-
  identical frames land on both sides.
- The evaluation reported here uses a **temporal holdout** instead: frames are
  ordered by position in the source video and divided into contiguous blocks, with a
  buffer discarded between blocks so that no evaluation frame falls within
  approximately 30 seconds of a training frame. The exact ranges are given in
  `configs/split_temporal.json` and in the README.

The temporal holdout removes the near-duplicate overlap between splits. It does not
remove the more fundamental limitation: training and evaluation frames still share a
single camera position, a single background, a single illumination regime, and in
most cases the same individual bees. The resulting figure describes performance
**on later parts of the same recording**, not performance on new footage.

## 3. Generalization has not been measured

No claim about generalization to other days, sites, species, or lighting conditions
is supported by the data in this repository. Establishing that requires annotating
held-out video drawn from different recording sessions and evaluating against it.
The 2025 season produced roughly 200 hours of footage across four sites, three daily
shifts, and nine weeks, so the material exists; the annotation does not yet. This is
the primary planned extension of the work.

## 4. The two evaluations differ in more than one respect

It is tempting to read the change from mAP50 0.745 to 0.632 as the size of the error
introduced by the split overlap. It is not, because two things changed at once.

The original run trained on **122 frames**. The temporal-holdout run trains on **79** —
the annotated pool is the same 126 frames, but assigning contiguous blocks to validation
and test, and discarding 13 frames to the buffer zones, leaves a smaller training set.
That is a **35% reduction in training data**, applied at the same time as the removal of
the leakage.

The observed difference therefore combines:

1. removal of the split leakage, which should lower the score because the model is no
   longer being scored partly on frames it memorised; and
2. a 35% smaller training set, which should also lower the score, for an unrelated
   reason.

**These two effects are not separable from this experiment**, and no attempt is made
here to attribute the difference between them. Isolating them would require retraining
on a training set of the original size with clean, disjoint assignment — which this
dataset cannot support, since the 126 annotated frames do not permit a 122-frame
training set and a disjoint temporal holdout simultaneously. It would need more
annotated frames, ideally from additional recordings.

The 0.632 figure should be read as *the performance of a model trained on 79 frames and
evaluated honestly*, not as *the corrected value of 0.745*.

## 5. The evaluation set is small, so the metric is imprecise

The temporal holdout validation set contains **23 frames and 38 bounding boxes**. The
metric computed on it is unbiased — the model never saw those frames during training —
but it is imprecise. With a sample this size, a small number of boxes changing from
detected to missed moves mAP substantially, and the confidence interval around any
reported value is correspondingly wide. The held-out test set is smaller still, at 11
frames and 11 boxes.

**The reported figures should be read as an order-of-magnitude check on whether the
detector works, not as a precise performance estimate.** They are sufficient to
establish that the model detects bees well above chance on frames it has not seen, and
not sufficient to support fine comparisons between models or claims about a specific
accuracy level.

A related observation supports treating this model as provisional: its validation mAP50
falls from 0.632 to 0.146 when the input is letterboxed to a square 640×640 rather than
batched at the native 16:9 aspect ratio. The `v2` checkpoint is essentially unaffected by
the same change. A model whose score depends that strongly on a benign preprocessing
choice is not robustly converged, which is consistent with it having been trained on 79
frames.

(For completeness: `v2` scores mAP50 0.951 on this same validation block. That number is
meaningless as a performance measure — every one of those 23 frames was in `v2`'s
training set, which is precisely the overlap described in section 1. It is reported here
only as the stability comparison above, and nowhere else.)

Bees occupy roughly 1–2% of frame width, placing this in the small-object detection
regime where mAP is particularly sensitive to a handful of boxes.

## 6. Synthetic annotation is proposed, not implemented

The motivating research question — whether synthetically composited training data
can substitute for manual annotation — remains open. **No synthetic data was
generated, no model was trained on synthetic data, and no manual-versus-synthetic
comparison was run.** The synthetic arm is described in this repository as planned
work only. Any figure describing a synthetic condition would be a placeholder, and
none is reported.
