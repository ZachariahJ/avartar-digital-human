"""Composites a generated mouth back into its driving frame.

Replaces MuseTalk's get_image_blending, which converts the whole 720x720 frame
to a PIL RGB Image in order to composite a 222x302 face box. The negative-stride
view it hands Pillow cannot be taken as a buffer, so fromarray falls back to a
full-frame tobytes() copy -- 8 of its 11ms per frame. Channel order never
mattered here: the blend is per-channel arithmetic, so it can stay in BGR.

Pixels are identical, not approximate. PIL composites an "L" mask as
dst + (src - dst) * a // 255 with a +127 bias and floor division, which is what
the expression below spells out. Outside the face box PIL blends the crop
against itself -- the identity -- so skipping it costs nothing.
"""
import numpy as np


def paste_face(frame, face, face_box, mask, crop_box) -> None:
    """Blend `face` into `frame` in place through `mask`'s alpha.

    Args:
        frame: driving frame, BGR uint8, modified in place -- pass a copy, the
            material's frames are shared by every render.
        face: generated mouth, BGR uint8, already resized to the face box.
        face_box: (x1, y1, x2, y2) of the face within `frame`.
        mask: cached blend mask, 8-bit, in crop_box coordinates.
        crop_box: the box `mask` was built against.
    """
    x1, y1, x2, y2 = face_box
    cx, cy = crop_box[0], crop_box[1]
    dst = frame[y1:y2, x1:x2]
    alpha = mask[y1 - cy:y2 - cy, x1 - cx:x2 - cx, None]
    # 尺寸不符说明素材错配
    if alpha.shape[:2] != dst.shape[:2] or face.shape != dst.shape:
        raise ValueError(f"blend geometry mismatch: face_box={face_box} "
                         f"crop_box={crop_box} mask={mask.shape}")
    # int32 必需：(s-d)*m 跨 ±65025
    alpha = alpha.astype(np.int32)
    frame[y1:y2, x1:x2] = (
        dst + ((face.astype(np.int32) - dst) * alpha + 127) // 255
    ).astype(np.uint8)
