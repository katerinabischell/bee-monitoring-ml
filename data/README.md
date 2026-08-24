# Data: description and provenance

**No image, video, or label data is stored in this repository.** This file describes
what the dataset is, how it was produced, and where to obtain it. Everything listed
under "Archived separately" below is deposited in the Zenodo release that accompanies
this repository; the DOI is recorded in the top-level README once minted.

## Provenance

All data derives from camera-trap video recorded by the author during the 2025 field
season at the North Campus Open Space (NCOS) restoration site, Goleta, California,
under the supervision of Dr. Katja Seltmann and Dr. Chris Evelyn at the UCSB Cheadle
Center for Biodiversity and Ecological Restoration.

Recording covered two federally endangered plant species — Ventura marsh milk-vetch
(*Astragalus pycnostachyus* var. *lanosissimus*) and salt marsh bird's-beak
(*Chloropyron maritimum* ssp. *maritimum*) — across nine weeks, four sites, and three
daily shifts, totalling approximately 200 hours of footage. Field protocol is
documented in [`../docs/methods.md`](../docs/methods.md).

Precise site coordinates are withheld. Site-level identifiers (sites 1–4) appear
throughout the data but are not georeferenced.

## What is in this repository

| Path | Contents |
|---|---|
| `../docs/field_log.csv` | The hand-kept session log, 86 rows. Date, week, site, camera, start/end time, weather, wind, observed activity, duration, shift, notes. Covers both species. |
| `../configs/dataset.yaml` | YOLO dataset definition (relative paths, single class `bee`). |
| `../configs/split_temporal.json` | The exact temporal holdout: cut points, buffer width, and the full frame list for each split. Sufficient to reconstruct the split from the archived frames. |
| `../results/metrics/` | Per-epoch training logs for all six runs, detection metadata, verification summaries. |
| `../results/detections/`, `../results/figures/` | A sample of annotated detection frames and annotation overlays, for illustration. |
| `../models/` | The published `ventura_bee_detection_v2` best checkpoint (21 MB). |

## The annotated dataset

Not distributed here; see the Zenodo archive.

- **Source.** A single 21-minute recording, `P1000471.MP4` (Week 6, Day 1, Site 1,
  morning shift), sampled at 1–2 second intervals.
- **Size.** 126 annotated frames containing 157 bounding boxes, single class `bee`.
- **Format.** YOLO text labels (`class cx cy w h`, normalised) alongside JPEG frames.
- **Box scale.** Bees occupy roughly 1–2% of frame width — small-object detection.
- **Splits.** A temporal holdout of 79 train / 23 validation / 11 test, with 13 frames
  discarded to buffer zones. Defined exactly in `configs/split_temporal.json`.

To reproduce the pipeline, download the frame archive from Zenodo and unpack it to
`data/yolo_dataset/` so that `train/images`, `val/images`, and `test/images` sit where
`configs/dataset.yaml` expects them.

Raw video is referenced by the extraction and inference scripts through the
`NCOS_VIDEO_ROOT` environment variable; point it at a local copy of the season
archive.

## Archived separately (Zenodo, not git)

These are excluded from version control for size or redundancy, not because they are
unavailable. A reader who wants them should look in the Zenodo deposit.

| Item | Size / count | Why excluded |
|---|---|---|
| `output_tracked.mp4` | 5.39 GB | Single rendered tracking demonstration video. Far beyond what git should carry. |
| Raw season footage | ~200 hours | Not redistributable at this scale; held by the Cheadle Center. |
| Annotated frame set | 126 JPEG frames + 126 label files | Distributed as one archive rather than 250+ tracked files. |
| Annotation overlay renders | 133 JPEG | Derived artifact, regenerable with `src/visualize_annotations.py`. |
| Full detection frame set | 313 JPEG | A representative 12 are committed under `results/detections/`. |
| Intermediate checkpoints | 29 `epoch*.pt` files, 64 MB each | Only the published best checkpoint is committed. |
| Other run checkpoints | 12 further `.pt` files | Superseded; metrics for every run are committed as CSV. |
| Full `runs/` directories | 4 validation runs | Ultralytics scratch output; the metrics that matter are in `results/metrics/`. |

## Excluded entirely

- **Grid-centre placeholder annotations** (`annotations_P1000474/`, 89 boxes, all
  exactly 0.2 × 0.2 at nine fixed positions). These are scaffolding produced from the
  observation log, never refined into real boxes, and never trained on. Publishing
  them alongside genuine annotations would invite them to be mistaken for ground
  truth.
- **A superseded dataset config** that pointed train, validation, and test at the same
  directory. It was not used by any training run and has been removed rather than
  shipped.
