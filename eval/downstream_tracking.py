"""
Downstream validation: does the interpolated middle frame place cloud
centroids where they actually are, better than the baseline does?

This is deliberately simple by design (per the 2-day scope cut) --
the sophistication is in what's being validated (tracking accuracy),
not in the tracker itself.

Method:
1. Detect cold cloud-top blobs in frame0 and frame2 (connected components
   on thresholded brightness temperature -- cold = convective cloud top).
2. Match blobs between frame0 and frame2 by nearest centroid (same cloud
   object, persisting across the ~10 min gap).
3. For each matched cloud, we know its REAL position at t+1 (from the
   ground-truth frame1, which you have in every test triplet).
4. Compare: how close is the cloud's centroid in (a) the baseline warped
   frame, (b) your model's predicted frame, to its TRUE position in the
   real frame1? Lower error = better tracking-relevant interpolation.

This produces one clean number per method: mean centroid position error
in pixels. That's your headline "downstream validation" result.
"""

import numpy as np
import cv2


def detect_cloud_centroids(frame_band13: np.ndarray, bt_threshold: float = 0.35, min_area: int = 20):
    """
    frame_band13: single-channel, normalized [0,1] Band 13 (higher = warmer).
    bt_threshold: pixels BELOW this normalized value are considered cold
        cloud tops. Default 0.35 corresponds to roughly 180 + 0.35*(320-180)
        = ~229K with the BT13_MIN/MAX=180/320 normalization used in
        build_triplets.py -- a reasonable convective cloud-top cutoff.
        Adjust if your data's cloud tops don't trigger enough detections.
    min_area: minimum blob size in pixels to count as a tracked cloud
        (filters noise/tiny specks).

    Returns: list of (x, y) centroids, and the labeled mask (for visualization).
    """
    cold_mask = (frame_band13 < bt_threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cold_mask, connectivity=8)

    valid_centroids = []
    for label_id in range(1, num_labels):  # label 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_centroids.append(tuple(centroids[label_id]))  # (x, y)

    return valid_centroids, labels


def match_centroids(centroids_a: list, centroids_b: list, max_dist: float = 40.0):
    """
    Greedy nearest-neighbor matching between two centroid lists.
    max_dist: pixels -- matches beyond this are rejected (not the same cloud,
        or the cloud dissipated/a new one formed -- don't force a match).

    Returns: list of (idx_a, idx_b, distance) for matched pairs.
    """
    if not centroids_a or not centroids_b:
        return []

    a = np.array(centroids_a)
    b = np.array(centroids_b)

    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)  # [len(a), len(b)]

    matches = []
    used_b = set()
    for i in range(len(a)):
        j = np.argmin(dists[i])
        if j not in used_b and dists[i, j] <= max_dist:
            matches.append((i, j, dists[i, j]))
            used_b.add(j)

    return matches


def evaluate_tracking_error(frame0_b13: np.ndarray, true_frame1_b13: np.ndarray,
                             pred_frame1_b13: np.ndarray, frame2_b13: np.ndarray,
                             bt_threshold: float = 0.35, min_area: int = 20,
                             max_match_dist: float = 40.0) -> dict:
    """
    Core metric for one triplet. Returns mean centroid position error
    (in pixels) between where clouds truly are in frame1 vs. where the
    predicted interpolated frame1 places them.

    frame0_b13, true_frame1_b13, pred_frame1_b13, frame2_b13: all single-channel
        Band 13, normalized [0,1]. pred_frame1_b13 can be your model's output
        OR the baseline's warped output -- run this function once per method
        and compare the returned errors.
    """
    c0, _ = detect_cloud_centroids(frame0_b13, bt_threshold, min_area)
    c_true1, _ = detect_cloud_centroids(true_frame1_b13, bt_threshold, min_area)
    c_pred1, _ = detect_cloud_centroids(pred_frame1_b13, bt_threshold, min_area)
    c2, _ = detect_cloud_centroids(frame2_b13, bt_threshold, min_area)

    # persistent clouds: present in BOTH frame0 and frame2 (survived the whole window)
    persistent_matches = match_centroids(c0, c2, max_match_dist)

    if not persistent_matches:
        return {"n_tracked_clouds": 0, "mean_error_px": None}

    errors = []
    for i0, i2, _ in persistent_matches:
        # expected position: nearest true detection in frame1 to the
        # linear midpoint of frame0/frame2 positions for this cloud
        mid_expected = (np.array(c0[i0]) + np.array(c2[i2])) / 2.0

        if not c_true1:
            continue
        true1_arr = np.array(c_true1)
        true_idx = np.argmin(np.linalg.norm(true1_arr - mid_expected, axis=1))
        true_pos = true1_arr[true_idx]

        if not c_pred1:
            continue
        pred1_arr = np.array(c_pred1)
        pred_idx = np.argmin(np.linalg.norm(pred1_arr - mid_expected, axis=1))
        pred_pos = pred1_arr[pred_idx]

        error = np.linalg.norm(true_pos - pred_pos)
        errors.append(error)

    if not errors:
        return {"n_tracked_clouds": len(persistent_matches), "mean_error_px": None}

    return {
        "n_tracked_clouds": len(persistent_matches),
        "mean_error_px": float(np.mean(errors)),
        "std_error_px": float(np.std(errors)),
    }


if __name__ == "__main__":
    # smoke test with synthetic data -- a blob moving diagonally
    h, w = 256, 256
    frame0 = np.ones((h, w), dtype=np.float32) * 0.7  # warm background
    frame2 = np.ones((h, w), dtype=np.float32) * 0.7
    true_frame1 = np.ones((h, w), dtype=np.float32) * 0.7
    pred_frame1_good = np.ones((h, w), dtype=np.float32) * 0.7
    pred_frame1_bad = np.ones((h, w), dtype=np.float32) * 0.7

    # cold blob at (50,50) in frame0, (70,70) in frame2 -> true midpoint ~(60,60)
    frame0[40:60, 40:60] = 0.1
    frame2[60:80, 60:80] = 0.1
    true_frame1[50:70, 50:70] = 0.1          # correct midpoint
    pred_frame1_good[52:72, 52:72] = 0.1     # close to correct
    pred_frame1_bad[20:40, 20:40] = 0.1      # way off

    result_good = evaluate_tracking_error(frame0, true_frame1, pred_frame1_good, frame2)
    result_bad = evaluate_tracking_error(frame0, true_frame1, pred_frame1_bad, frame2)

    print("Good prediction result:", result_good)
    print("Bad prediction result:", result_bad)
    assert result_good["mean_error_px"] < result_bad["mean_error_px"], "Sanity check failed!"
    print("\nSmoke test passed: good prediction scores lower error than bad prediction.")