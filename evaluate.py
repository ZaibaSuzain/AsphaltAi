"""
evaluate.py - measure how good the pothole detector really is on YOUR labeled photos.

A confidence of 91% only means the model FELT 91% sure. It does not mean it is right 91% of
the time. This script compares the model's answers with boxes a human drew (the "ground truth")
and reports: precision, recall, F1, AP50, how reliable the confidence numbers are,
results per photo type, speed, and review pictures that let you SEE each mistake.

Folder layout (names must match, e.g. rain_01.jpg and rain_01.txt):
    test_data/images/rain_01.jpg
    test_data/labels/rain_01.txt     <- YOLO format, one line per pothole: 0 cx cy w h

Photos with NO potholes need no label file (or an empty one).
The part of the file name before the first underscore is the photo's GROUP
(normal, multi, none, rain, dark, shadow, patched ...). Results are also shown per group.

Run (from the project folder, with the venv active):
    python evaluate.py --selftest     checks the maths with made-up boxes, no photos needed
    python evaluate.py                real evaluation
    python evaluate.py --draft-labels the model drafts label files for you to check and fix
"""

import argparse
import math
import statistics
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
IOU_MATCH = 0.5   # a detection is "correct" if it overlaps a real pothole box by at least 50%
LOW_FLOOR = 0.01  # ask the model for everything, so every possible cutoff can be tested
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 ... 0.9
CALIBRATION_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
OUT_DIR = Path("eval_results")
REVIEW_MIN = 0.10  # review pictures hide model detections below this (they are mostly noise)


# ----------------------------------------------------------------------
# 1. BOXES: overlap, reading the human labels, grouping
# ----------------------------------------------------------------------
def iou(a, b):
    """Overlap between two boxes [x1, y1, x2, y2]: 0 = no overlap, 1 = identical."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_labels(label_path, width, height):
    """Read a YOLO label file and convert it to pixel boxes [x1, y1, x2, y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{label_path.name} is not in YOLO box format (class cx cy w h). "
                "Export the annotations again as 'YOLO' boxes, not polygons."
            )
        cx, cy, bw, bh = (float(p) for p in parts[1:5])
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def group_of(path):
    stem = path.stem
    return stem.split("_")[0].lower() if "_" in stem else "ungrouped"


# ----------------------------------------------------------------------
# 2. SCORING: which detections were right, and the summary numbers
# ----------------------------------------------------------------------
def match_photo(gt_boxes, photo_preds):
    """
    Score ONE photo. Goes through the model's detections from most to least confident.
    A detection is correct if it overlaps a not-yet-used real pothole by IOU_MATCH or more;
    each real pothole can be claimed only once.
    Returns a list of (confidence, box, is_correct).
    """
    used = set()
    scored = []
    for conf, box in sorted(photo_preds, key=lambda p: p[0], reverse=True):
        best_iou, best_j = 0.0, None
        for j, gt_box in enumerate(gt_boxes):
            if j in used:
                continue
            overlap = iou(box, gt_box)
            if overlap > best_iou:
                best_iou, best_j = overlap, j
        is_correct = best_j is not None and best_iou >= IOU_MATCH
        if is_correct:
            used.add(best_j)
        scored.append((conf, box, is_correct))
    return scored


def match_predictions(gts, preds):
    """
    gts[i]   = real pothole boxes in photo i
    preds[i] = list of (confidence, box) the model produced for photo i
    Returns a list of (confidence, is_correct, photo_index), most confident first.
    """
    results = []
    for i, (gt_boxes, photo_preds) in enumerate(zip(gts, preds)):
        for conf, _, is_correct in match_photo(gt_boxes, photo_preds):
            results.append((conf, is_correct, i))
    results.sort(key=lambda r: r[0], reverse=True)
    return results


def prf(tp, fp, total_gt):
    """
    precision = of everything the model flagged, how much was a real pothole
    recall    = of all real potholes, how many the model found
    F1        = one number balancing both
    """
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / total_gt if total_gt else math.nan
    if math.isnan(recall):
        f1 = math.nan
    elif math.isnan(precision) or precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def pr_curve(results, total_gt):
    """Precision and recall as the cutoff slides from strict to loose."""
    recalls, precisions = [0.0], [1.0]
    tp = fp = 0
    for _, is_correct, _ in results:
        if is_correct:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / total_gt)
        precisions.append(tp / (tp + fp))
    return recalls, precisions


def average_precision(results, total_gt):
    """AP50: area under the precision-recall curve. It does not depend on any one cutoff."""
    if total_gt == 0 or not results:
        return 0.0
    recalls, precisions = pr_curve(results, total_gt)
    for i in range(len(precisions) - 2, -1, -1):          # smooth the curve (standard step)
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in range(1, len(recalls)))


# ----------------------------------------------------------------------
# 3. TABLES
# ----------------------------------------------------------------------
def threshold_table(results, total_gt):
    rows = []
    for t in THRESHOLDS:
        kept = [ok for conf, ok, _ in results if conf >= t]
        tp = sum(kept)
        fp = len(kept) - tp
        precision, recall, f1 = prf(tp, fp, total_gt)
        rows.append({"confidence cutoff": t, "TP": tp, "FP": fp, "FN": total_gt - tp,
                     "precision": precision, "recall": recall, "F1": f1})
    return pd.DataFrame(rows)


def group_table(results, gts, groups, t):
    rows = []
    for g in sorted(set(groups)):
        idx = {i for i, name in enumerate(groups) if name == g}
        kept = [ok for conf, ok, i in results if conf >= t and i in idx]
        gt_count = sum(len(gts[i]) for i in idx)
        tp = sum(kept)
        fp = len(kept) - tp
        precision, recall, f1 = prf(tp, fp, gt_count)
        rows.append({"group": g, "photos": len(idx), "real potholes": gt_count,
                     "TP": tp, "FP": fp, "FN": gt_count - tp,
                     "precision": precision, "recall": recall, "F1": f1})
    return pd.DataFrame(rows)


def calibration_table(results):
    """When the model says X% sure, how often is it actually right?"""
    rows = []
    for low, high in CALIBRATION_BINS:
        in_bin = [ok for conf, ok, _ in results if low <= conf < high]
        correct = sum(in_bin)
        rows.append({"model confidence": f"{low:.0%}-{min(high, 1.0):.0%}",
                     "detections": len(in_bin),
                     "actually correct": correct,
                     "% actually correct": correct / len(in_bin) * 100 if in_bin else math.nan})
    return pd.DataFrame(rows)


def show(title, df):
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def save_pr_curve(results, total_gt, path):
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    recalls, precisions = pr_curve(results, total_gt)
    plt.figure(figsize=(5, 4))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.title("Precision-recall curve (IoU 0.5)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# 4. RUNNING THE MODEL
# ----------------------------------------------------------------------
def run_model(model, image_paths, imgsz):
    """Run the model on every photo. Returns detections, time per photo (ms), photo sizes."""
    def load(path):
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

    model.predict(load(image_paths[0]), conf=LOW_FLOOR, imgsz=imgsz, verbose=False)  # warm-up
    preds, times, sizes = [], [], []
    for path in image_paths:
        image = load(path)
        start = time.perf_counter()
        result = model.predict(image, conf=LOW_FLOOR, imgsz=imgsz, verbose=False)[0]
        times.append((time.perf_counter() - start) * 1000)
        preds.append(list(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist())))
        sizes.append(image.size)
    return preds, times, sizes


def evaluate_size(model, image_paths, labels_dir, imgsz):
    preds, times, sizes = run_model(model, image_paths, imgsz)
    gts = [read_labels(labels_dir / f"{p.stem}.txt", w, h)
           for p, (w, h) in zip(image_paths, sizes)]
    return {"results": match_predictions(gts, preds), "gts": gts, "times": times,
            "preds": preds}


def save_review_images(image_paths, gts, preds, cutoff, out_dir):
    """
    Save a copy of every photo with boxes drawn on it, so you can SEE each result:
      green  = the potholes YOU labeled
      red    = model detections at or above the cutoff ("OK" = matched one of your boxes)
      orange = model detections between REVIEW_MIN and the cutoff (ignored by the tables)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, gt_boxes, photo_preds in zip(image_paths, gts, preds):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        draw = ImageDraw.Draw(image)
        line = max(2, image.width // 200)
        size = max(14, image.width // 35)
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:  # very old Pillow without a font size option
            font = ImageFont.load_default()

        for box in gt_boxes:
            draw.rectangle(box, outline="#00cc44", width=line)
        for conf, box, is_correct in match_photo(gt_boxes, photo_preds):
            if conf < REVIEW_MIN:
                continue
            counted = conf >= cutoff
            color = "#ee2222" if counted else "#ff9900"
            draw.rectangle(box, outline=color, width=line)
            label = f"{conf:.0%}" + (" OK" if counted and is_correct else "")
            draw.text((box[0] + 3, max(0, box[1] - size - 2)), label, fill=color, font=font,
                      stroke_width=2, stroke_fill="black")

        legend = [("green = your labels", "#00cc44"),
                  (f"red = model, {cutoff:.0%} or more", "#ee2222"),
                  (f"orange = model, {REVIEW_MIN:.0%}-{cutoff:.0%} (ignored)", "#ff9900")]
        for n, (text, color) in enumerate(legend):
            draw.text((5, 5 + n * (size + 4)), text, fill=color, font=font,
                      stroke_width=2, stroke_fill="black")
        image.save(out_dir / f"{path.stem}.jpg", quality=90)


def write_draft_labels(model, image_paths, out_dir, cutoff, imgsz):
    """
    Let the model PROPOSE boxes and save them as YOLO label files (a first draft).
    These are NOT the answer key until a human has checked every photo and fixed them.
    Photos where the model proposes nothing get no file (= 'no potholes').
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, _, sizes = run_model(model, image_paths, imgsz)
    written = 0
    for path, photo_preds, (w, h) in zip(image_paths, preds, sizes):
        lines = [f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                 f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                 for conf, (x1, y1, x2, y2) in photo_preds if conf >= cutoff]
        if lines:
            (out_dir / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")
            written += 1
    (out_dir / "labels.txt").write_text("pothole\n")  # the class name list some tools ask for
    return written


# ----------------------------------------------------------------------
# 5. SELF-TEST: checks the maths using tiny made-up boxes (not your model)
# ----------------------------------------------------------------------
def run_selftest():
    gts = [[[0, 0, 100, 100]], []]                    # photo 0 has one pothole, photo 1 has none
    preds = [[(0.9, [0, 0, 100, 100]), (0.8, [200, 200, 300, 300])],   # one right, one wrong
             [(0.7, [10, 10, 50, 50])]]                                # a false alarm
    results = match_predictions(gts, preds)
    tp = sum(ok for _, ok, _ in results)
    assert tp == 1 and len(results) == 3
    precision, recall, _ = prf(tp=1, fp=2, total_gt=1)
    assert abs(precision - 1 / 3) < 1e-9 and recall == 1.0
    assert abs(average_precision(results, 1) - 1.0) < 1e-9
    assert average_precision(match_predictions([[[0, 0, 10, 10]]], [[]]), 1) == 0.0  # all missed
    print("Self-test passed: the matching, precision, recall and AP maths behave correctly.")


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate the pothole detector on labeled photos.")
    parser.add_argument("--images", default="test_data/images")
    parser.add_argument("--labels", default="test_data/labels")
    parser.add_argument("--weights", default="models/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="confidence cutoff used for the per-group table (default 0.5)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 320],
                        help="image sizes to compare for speed; the first one gets the full report")
    parser.add_argument("--draft-labels", action="store_true",
                        help="let the model draft label files into test_data/labels_draft, then stop")
    parser.add_argument("--draft-cutoff", type=float, default=0.30,
                        help="minimum confidence for a drafted box (default 0.30)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    images_dir, labels_dir = Path(args.images), Path(args.labels)
    if not images_dir.exists():
        raise SystemExit(f"Image folder not found: {images_dir}\n"
                         "Create it and add your test photos (see the top of this file).")
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"No JPG or PNG photos found in {images_dir}.")
    if not Path(args.weights).exists():
        raise SystemExit(f"Model weights not found at {args.weights}. "
                         "Run the Streamlit app once so it downloads them.")

    missing = [p for p in image_paths if not (labels_dir / f"{p.stem}.txt").exists()]
    if missing:
        print(f"{len(missing)} photos have no label file and are counted as having NO potholes.")
        suspicious = [p.name for p in missing if group_of(p) != "none"]
        if suspicious:
            print("Check these. If they do contain potholes, annotate them first:")
            print("  " + ", ".join(suspicious))

    from ultralytics import YOLO
    model = YOLO(args.weights)

    if args.draft_labels:
        draft_dir = labels_dir.parent / "labels_draft"
        count = write_draft_labels(model, image_paths, draft_dir, args.draft_cutoff, args.sizes[0])
        print(f"\nDrafted boxes for {count} of {len(image_paths)} photos in '{draft_dir}'.")
        print("These are only the model's guesses. Load them into your labeling tool, check EVERY "
              "photo, fix wrong, loose or missing boxes, export, and put the final .txt files in "
              f"'{labels_dir}'. Do not evaluate on the draft itself: that would mark the model "
              "with its own answers.")
        return

    groups = [group_of(p) for p in image_paths]
    OUT_DIR.mkdir(exist_ok=True)

    speed_rows = []
    for k, size in enumerate(args.sizes):
        print(f"\nRunning the model on {len(image_paths)} photos at image size {size} ...")
        ev = evaluate_size(model, image_paths, labels_dir, size)
        total_gt = sum(len(g) for g in ev["gts"])
        if total_gt == 0:
            raise SystemExit("No labeled potholes found. Check that your label files are in "
                             f"{labels_dir} and match the photo names.")

        table = threshold_table(ev["results"], total_gt)
        ap = average_precision(ev["results"], total_gt)
        valid = table.dropna(subset=["F1"])
        best = valid.loc[valid["F1"].idxmax()] if not valid.empty else None
        speed_rows.append({
            "image size": size,
            "mean ms": statistics.mean(ev["times"]),
            "median ms": statistics.median(ev["times"]),
            "AP50": ap,
            "best F1": best["F1"] if best is not None else math.nan,
            "at cutoff": best["confidence cutoff"] if best is not None else math.nan,
        })

        if k == 0:  # full report for the first (main) image size
            print(f"\n=== MAIN RESULTS (image size {size}) ===")
            print(f"{len(image_paths)} photos, {total_gt} real potholes labeled, AP50 = {ap:.3f}")
            show("Precision / recall / F1 at different confidence cutoffs:", table)
            show(f"Results per group at cutoff {args.threshold}:",
                 group_table(ev["results"], ev["gts"], groups, args.threshold))
            calibration = calibration_table(ev["results"])
            show("Is the confidence number trustworthy? (model confidence vs. how often right):",
                 calibration)
            table.to_csv(OUT_DIR / "thresholds.csv", index=False)
            group_table(ev["results"], ev["gts"], groups, args.threshold).to_csv(
                OUT_DIR / "groups.csv", index=False)
            calibration.to_csv(OUT_DIR / "calibration.csv", index=False)
            save_pr_curve(ev["results"], total_gt, OUT_DIR / "pr_curve.png")
            save_review_images(image_paths, ev["gts"], ev["preds"], args.threshold,
                               OUT_DIR / "review")
            print(f"\nReview pictures saved in '{OUT_DIR / 'review'}': green = your boxes, "
                  f"red = model at {args.threshold:.0%} or more, orange = model below that.")

    speed = pd.DataFrame(speed_rows)
    show("Speed vs. accuracy for different image sizes:", speed)
    speed.to_csv(OUT_DIR / "speed.csv", index=False)

    print(f"\nSaved tables and pr_curve.png in the '{OUT_DIR}' folder.")
    print("Remember: these numbers only describe YOUR test photos. Always say how many photos "
          "you used, where they came from, and which cutoff you reported.")


if __name__ == "__main__":
    main()