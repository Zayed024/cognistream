"""
CogniStream — Set-of-Mark (SoM) Overlay

Draws numbered bounding boxes on a frame so the VLM can reference
objects by ID in its output (e.g., "person_3 hands bag to person_7").

Borrowed from NVIDIA Metropolis VSS 3 — eliminates the "which one
did the VLM mean" ambiguity in scenes with multiple similar objects.

How it works:
    1. CV pre-filter detects objects in the frame
    2. Each detection gets a numeric ID
    3. cv2 draws a box + ID label on the frame
    4. Annotated frame is sent to the VLM
    5. VLM output references objects by ID

The VLM caption can be parsed to map IDs back to detections.

Usage:
    from backend.visual.som_overlay import draw_som
    from backend.visual.cv_filter import cv_filter

    detections = cv_filter.detect("frame.jpg")
    annotated_path = draw_som("frame.jpg", detections, "frame_som.jpg")
    # Send annotated_path to VLM
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from backend.visual.cv_filter import Detection

logger = logging.getLogger(__name__)


# Color palette for box overlays — distinct colors for IDs 0-9
PALETTE = [
    (66, 135, 245),   # blue
    (245, 135, 66),   # orange
    (66, 245, 135),   # green
    (245, 66, 135),   # pink
    (135, 66, 245),   # purple
    (245, 245, 66),   # yellow
    (66, 245, 245),   # cyan
    (245, 66, 66),    # red
    (135, 245, 66),   # lime
    (66, 66, 245),    # navy
]


def draw_som(
    image_path: str,
    detections: list[Detection],
    output_path: str | None = None,
    box_thickness: int = 3,
    label_scale: float = 0.7,
    show_confidence: bool = False,
) -> str:
    """Draw numbered bounding boxes on an image for SoM prompting.

    Args:
        image_path: Source image.
        detections: List of Detection objects from CV filter.
        output_path: Where to save the annotated image. If None, overwrites input.
        box_thickness: Box line thickness in pixels.
        label_scale: Font scale for ID labels.
        show_confidence: If True, includes confidence in the label.

    Returns:
        Path to the annotated image.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Cannot read image for SoM: %s", image_path)
        return image_path

    if not detections:
        # No detections — return original
        if output_path:
            cv2.imwrite(output_path, img)
            return output_path
        return image_path

    # Assign track_ids if missing
    for i, det in enumerate(detections):
        if det.track_id < 0:
            det.track_id = i + 1

    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        # Clip to image bounds
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        color = PALETTE[(det.track_id - 1) % len(PALETTE)]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        # Build label: "1: person" or "1: person 0.85"
        if show_confidence:
            label = f"{det.track_id}: {det.label} {det.confidence:.2f}"
        else:
            label = f"{det.track_id}: {det.label}"

        # Draw label background + text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, label_scale, 2)
        # Background rectangle for text
        label_y = y1 - 6 if y1 > th + 10 else y1 + th + 6
        cv2.rectangle(
            img,
            (x1, label_y - th - 4),
            (x1 + tw + 6, label_y + baseline),
            color,
            -1,  # filled
        )
        # Text in white
        cv2.putText(
            img, label,
            (x1 + 3, label_y - 2),
            font, label_scale, (255, 255, 255), 2,
            cv2.LINE_AA,
        )

    out = output_path or image_path
    cv2.imwrite(out, img)
    return out


def build_som_prompt(detections: list[Detection], base_prompt: str) -> str:
    """Augment a VLM prompt with the SoM ID legend.

    Args:
        detections: The detections that were drawn on the frame.
        base_prompt: The original VLM prompt (e.g., the combined prompt).

    Returns:
        A prompt with the ID-to-label legend prepended.
    """
    if not detections:
        return base_prompt

    legend_lines = ["Numbered objects in the image:"]
    for det in detections:
        legend_lines.append(f"  {det.track_id}: {det.label}")
    legend = "\n".join(legend_lines)

    return (
        f"{legend}\n\n"
        f"When you reference these objects in your answer, use their numbers "
        f"(e.g., 'person 1 hands bag to person 2').\n\n"
        f"{base_prompt}"
    )
