import numpy as np


def paste_face(frame, face, face_box, mask, crop_box) -> None:
    x1, y1, x2, y2 = face_box
    cx, cy = crop_box[0], crop_box[1]
    dst = frame[y1:y2, x1:x2]
    alpha = mask[y1 - cy:y2 - cy, x1 - cx:x2 - cx, None]
    if alpha.shape[:2] != dst.shape[:2] or face.shape != dst.shape:
        raise ValueError(f"blend geometry mismatch: face_box={face_box} "
                         f"crop_box={crop_box} mask={mask.shape}")
    alpha = alpha.astype(np.int32)
    frame[y1:y2, x1:x2] = (
        dst + ((face.astype(np.int32) - dst) * alpha + 127) // 255
    ).astype(np.uint8)
