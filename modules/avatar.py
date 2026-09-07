"""MuseTalk talking-head renderer.

Replaces FLOAT. The public API is deliberately unchanged — get_pool(),
generate_video(), generate_idle_video() — so modules/pipeline.py and main.py
call this exactly as before.

The model swap changes what the avatar IS driven by. FLOAT animated a single
still portrait and invented head motion from the audio. MuseTalk is an
inpainting model: it repaints ONLY the mouth region of an existing clip, so
every other pixel and all head motion comes from config.AVATAR_VIDEO. The
per-frame face boxes, VAE latents and blend masks of that clip are expensive to
compute, so they are prepared once into config.MUSETALK_MATERIAL_DIR (keyed by
config.avatar_fingerprint()) and reused by every render afterwards.
"""
import contextlib
import glob
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

import config

# Add MuseTalk to path
sys.path.insert(0, config.MUSETALK_DIR)

import torch

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()

_mt = None          # cached handles to MuseTalk's own functions
_mt_lock = threading.Lock()

# MuseTalk's marker for "no face found in this frame"
_NO_FACE = (0.0, 0.0, 0.0, 0.0)


@contextlib.contextmanager
def _cwd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _musetalk():
    """Import MuseTalk's modules once, with CWD == MUSETALK_DIR.

    The chdir is NOT optional. musetalk/utils/preprocessing.py builds the DWPose
    landmark model at IMPORT time from the relative paths
    './musetalk/utils/dwpose/rtmpose-l_...py' and
    './models/dwpose/dw-ll_ucoco_384.pth', and FaceParsing's BiSeNet checkpoint
    defaults are relative too. chdir is process-global, so it is confined to
    this one import (and to material preparation, which constructs FaceParsing);
    both run under the pool lock at startup, before the server serves anything.
    """
    global _mt
    if _mt is None:
        with _mt_lock:
            if _mt is None:
                t0 = time.perf_counter()
                with _cwd(config.MUSETALK_DIR):
                    from musetalk.utils.utils import load_all_model, datagen
                    from musetalk.utils.preprocessing import get_landmark_and_bbox
                    from musetalk.utils.blending import (
                        get_image_blending, get_image_prepare_material,
                    )
                    from musetalk.utils.face_parsing import FaceParsing
                    from musetalk.utils.audio_processor import AudioProcessor
                    from transformers import WhisperModel
                _mt = SimpleNamespace(
                    load_all_model=load_all_model,
                    datagen=datagen,
                    get_landmark_and_bbox=get_landmark_and_bbox,
                    get_image_blending=get_image_blending,
                    get_image_prepare_material=get_image_prepare_material,
                    FaceParsing=FaceParsing,
                    AudioProcessor=AudioProcessor,
                    WhisperModel=WhisperModel,
                )
                logger.info("MuseTalk imported in %.1fs", time.perf_counter() - t0)
    return _mt


# --------------- Driving material ---------------

class _Material:
    """One prepared cycle of driving frames.

    The cycle is the source clip forward THEN reversed, so playback wraps
    without a jump: frame N-1 is the neighbour of frame 0. Everything is
    per-frame aligned — frames[i], coords[i], latents[i], masks[i] and
    mask_coords[i] all describe the same frame.

    Frames are kept decoded in RAM (~3 bytes/pixel × 2× the source clip's
    frames), which is why AVATAR_VIDEO should stay a few seconds long.
    """

    __slots__ = ("frames", "coords", "latents", "masks", "mask_coords")

    def __init__(self, frames, coords, latents, masks, mask_coords):
        self.frames = frames
        self.coords = coords
        self.latents = latents
        self.masks = masks
        self.mask_coords = mask_coords

    def __len__(self):
        return len(self.frames)


def _material_dir() -> str:
    return os.path.join(config.MUSETALK_MATERIAL_DIR, config.avatar_fingerprint())


def _video_to_frames(video_path: str) -> list:
    """Decode the driving video to BGR frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open driving video {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"driving video {video_path} has no decodable frames")
    # The whisper->frame alignment assumes the material runs at MUSETALK_FPS. A
    # mismatch does not fail, it just drifts the lips, so say so loudly.
    if abs(src_fps - config.MUSETALK_FPS) > 0.5:
        logger.warning("driving video is %.2f fps but MUSETALK_FPS is %d — lip sync "
                       "will drift; re-encode the clip to %d fps",
                       src_fps, config.MUSETALK_FPS, config.MUSETALK_FPS)
    return frames


def _prepare_material(mat_dir: str, vae) -> _Material:
    """Run the one-off pass over AVATAR_VIDEO: face boxes, VAE latents, blend
    masks. Written to a scratch dir and renamed into place, so an interrupted
    preparation can never leave a half-built material dir that later loads.
    """
    mt = _musetalk()
    logger.info("Preparing MuseTalk driving material from %s (one-off, minutes)...",
                config.AVATAR_VIDEO)
    t0 = time.perf_counter()

    build_dir = mat_dir + ".building"
    shutil.rmtree(build_dir, ignore_errors=True)
    imgs_dir = os.path.join(build_dir, "full_imgs")
    masks_dir = os.path.join(build_dir, "mask")
    os.makedirs(imgs_dir)
    os.makedirs(masks_dir)

    # get_landmark_and_bbox reads image FILES, so the decoded frames go to disk
    # first (upstream does the same).
    src_frames = _video_to_frames(config.AVATAR_VIDEO)
    for i, frame in enumerate(src_frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
    img_list = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))

    coord_list, frame_list = mt.get_landmark_and_bbox(img_list, config.MUSETALK_BBOX_SHIFT)

    coords, frames, latents = [], [], []
    for bbox, frame in zip(coord_list, frame_list):
        # Drop frames with no detected face instead of keeping them. Upstream
        # skips only the latent, which silently shifts every later frame's
        # latent onto the wrong image; dropping the whole frame keeps the three
        # lists aligned.
        if tuple(bbox) == _NO_FACE:
            continue
        x1, y1, x2, y2 = bbox
        if config.MUSETALK_VERSION == "v15":
            y2 = min(y2 + config.MUSETALK_EXTRA_MARGIN, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(crop).cpu())
        coords.append([x1, y1, x2, y2])
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"no face detected in any frame of {config.AVATAR_VIDEO}")
    if len(frames) < len(frame_list):
        logger.warning("dropped %d of %d driving frames with no detected face",
                       len(frame_list) - len(frames), len(frame_list))

    # Forward + reversed = seamless loop.
    frames = frames + frames[::-1]
    coords = coords + coords[::-1]
    latents = latents + latents[::-1]

    mode = config.MUSETALK_PARSING_MODE if config.MUSETALK_VERSION == "v15" else "raw"
    with _cwd(config.MUSETALK_DIR):
        fp = mt.FaceParsing(
            left_cheek_width=config.MUSETALK_LEFT_CHEEK_WIDTH,
            right_cheek_width=config.MUSETALK_RIGHT_CHEEK_WIDTH,
        ) if config.MUSETALK_VERSION == "v15" else mt.FaceParsing()

    masks, mask_coords = [], []
    # The png files are rewritten in cycle order so a reload sees exactly the
    # frames prepared here, not the raw decode.
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
        mask, crop_box = mt.get_image_prepare_material(frame, coords[i], fp=fp, mode=mode)
        cv2.imwrite(os.path.join(masks_dir, f"{i:08d}.png"), mask)
        masks.append(mask)
        mask_coords.append(crop_box)
    # Any leftover pngs from the raw decode (when faces were dropped) would be
    # picked up by the glob on reload.
    for stale in sorted(glob.glob(os.path.join(imgs_dir, "*.png")))[len(frames):]:
        os.remove(stale)

    del fp
    torch.cuda.empty_cache()

    torch.save(latents, os.path.join(build_dir, "latents.pt"))
    with open(os.path.join(build_dir, "coords.pkl"), "wb") as f:
        pickle.dump(coords, f)
    with open(os.path.join(build_dir, "mask_coords.pkl"), "wb") as f:
        pickle.dump(mask_coords, f)
    with open(os.path.join(build_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video": config.AVATAR_VIDEO,
            "fingerprint": config.avatar_fingerprint(),
            "version": config.MUSETALK_VERSION,
            "fps": config.MUSETALK_FPS,
            "frames": len(frames),
        }, f)

    shutil.rmtree(mat_dir, ignore_errors=True)
    os.replace(build_dir, mat_dir)
    logger.info("Driving material ready: %d frames in %.1fs -> %s",
                len(frames), time.perf_counter() - t0, mat_dir)
    return _Material(frames, coords, latents, masks, mask_coords)


def _load_material(mat_dir: str) -> _Material:
    """Load material prepared by an earlier run."""
    def _imgs(sub):
        paths = sorted(glob.glob(os.path.join(mat_dir, sub, "*.png")))
        return [cv2.imread(p, cv2.IMREAD_UNCHANGED if sub == "mask" else cv2.IMREAD_COLOR)
                for p in paths]

    latents = torch.load(os.path.join(mat_dir, "latents.pt"),
                         map_location="cpu", weights_only=True)
    with open(os.path.join(mat_dir, "coords.pkl"), "rb") as f:
        coords = pickle.load(f)
    with open(os.path.join(mat_dir, "mask_coords.pkl"), "rb") as f:
        mask_coords = pickle.load(f)
    frames = _imgs("full_imgs")
    masks = _imgs("mask")
    n = min(len(frames), len(coords), len(latents), len(masks), len(mask_coords))
    if n == 0 or len({len(frames), len(coords), len(latents), len(masks), len(mask_coords)}) != 1:
        raise RuntimeError(f"material at {mat_dir} is inconsistent; delete it to rebuild")
    logger.info("Loaded MuseTalk driving material: %d frames from %s", n, mat_dir)
    return _Material(frames, coords, latents, masks, mask_coords)


def _get_material(vae) -> _Material:
    mat_dir = _material_dir()
    if os.path.exists(os.path.join(mat_dir, "latents.pt")):
        return _load_material(mat_dir)
    os.makedirs(config.MUSETALK_MATERIAL_DIR, exist_ok=True)
    return _prepare_material(mat_dir, vae)


# --------------- Rendering ---------------

def _run_h264_encode(base_cmd: list[str], output_path: str,
                     audio_path: str | None = None) -> str:
    """Finish an FFmpeg command using the available browser-safe H.264 encoder."""
    # Some server FFmpeg builds omit GPL libx264 but include OpenH264. Both
    # produce browser-compatible H.264, so use OpenH264 as a portable fallback.
    encoders = [
        ("libx264", ["-crf", "18"]),
        ("libopenh264", ["-b:v", "4M"]),
    ]
    errors = []
    for encoder, quality_args in encoders:
        cmd = base_cmd + ["-c:v", encoder, "-pix_fmt", "yuv420p"] + quality_args
        cmd += (["-c:a", "aac", "-b:a", "128k", "-shortest"]
                if audio_path else ["-an"])
        cmd += [output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if encoder != "libx264":
                logger.info("FFmpeg libx264 unavailable; encoded with %s", encoder)
            return output_path
        errors.append(f"{encoder}: {result.stderr.strip()}")
    raise RuntimeError("FFmpeg could not encode H.264: " + " | ".join(errors))


def _encode(frames_dir: str, output_path: str, audio_path: str | None) -> str:
    """Mux the rendered png sequence (and the spoken audio) into one mp4."""
    base_cmd = ["ffmpeg", "-y", "-v", "error",
                "-framerate", str(config.MUSETALK_FPS),
                "-i", os.path.join(frames_dir, "%08d.png")]
    if audio_path:
        base_cmd += ["-i", audio_path]
    return _run_h264_encode(base_cmd, output_path, audio_path)


def _generate_motion_idle(duration: float, output_path: str) -> str:
    """Keep natural head/eye motion while replacing speech with one smiling mouth.

    A short section around IDLE_SMILE_FRAME_SEC is played forward then backward,
    which makes the boundary seamless. MuseTalk's cached per-frame face boxes and
    soft jaw masks track one reference smile onto that motion without inference.
    """
    mat_dir = _material_dir()
    with open(os.path.join(mat_dir, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    with open(os.path.join(mat_dir, "coords.pkl"), "rb") as f:
        coords = pickle.load(f)
    with open(os.path.join(mat_dir, "mask_coords.pkl"), "rb") as f:
        mask_coords = pickle.load(f)

    # The cache is source frames followed by their reverse. Work only in its
    # first half, then construct our own ping-pong cycle without duplicate ends.
    source_count = int(info["frames"]) // 2
    fps = config.MUSETALK_FPS
    reference = min(source_count - 1, max(0, round(config.IDLE_SMILE_FRAME_SEC * fps)))
    radius = max(1, round(duration * fps / 4))
    start = max(0, reference - radius)
    end = min(source_count - 1, reference + radius)
    indices = list(range(start, end + 1)) + list(range(end - 1, start, -1))

    reference_frame = cv2.imread(
        os.path.join(mat_dir, "full_imgs", f"{reference:08d}.png"))
    if reference_frame is None:
        raise RuntimeError(f"missing cached idle reference frame {reference}")
    rx1, ry1, rx2, ry2 = coords[reference]
    reference_face = reference_frame[ry1:ry2, rx1:rx2]
    if reference_face.size == 0:
        raise RuntimeError(f"invalid cached idle reference face {coords[reference]}")

    frames_dir = tempfile.mkdtemp(prefix="idleframes_", dir=config.TEMP_DIR)
    try:
        for output_index, frame_index in enumerate(indices):
            frame = cv2.imread(os.path.join(
                mat_dir, "full_imgs", f"{frame_index:08d}.png"))
            mask = cv2.imread(os.path.join(
                mat_dir, "mask", f"{frame_index:08d}.png"), cv2.IMREAD_GRAYSCALE)
            if frame is None or mask is None:
                raise RuntimeError(f"missing cached idle material frame {frame_index}")

            x1, y1, x2, y2 = coords[frame_index]
            xs, ys, xe, ye = mask_coords[frame_index]
            smile = cv2.resize(reference_face, (x2 - x1, y2 - y1),
                               interpolation=cv2.INTER_LANCZOS4)
            replacement = frame[ys:ye, xs:xe].copy()
            replacement[y1 - ys:y2 - ys, x1 - xs:x2 - xs] = smile
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            target = frame[ys:ye, xs:xe]
            frame[ys:ye, xs:xe] = (
                replacement * alpha + target * (1.0 - alpha)
            ).astype(np.uint8)
            cv2.imwrite(os.path.join(frames_dir, f"{output_index:08d}.png"), frame)

        return _encode(frames_dir, output_path, audio_path=None)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


class _Worker:
    """One MuseTalk stack pinned to one GPU."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.material = None

    def load(self):
        mt = _musetalk()
        # load_all_model hardcodes the VAE path as os.path.join("models", vae_type).
        # Passing an ABSOLUTE vae_type makes that join return the absolute path
        # unchanged, which is how the VAE gets found without a chdir here.
        vae, unet, pe = mt.load_all_model(
            unet_model_path=config.MUSETALK_UNET,
            vae_type=config.MUSETALK_VAE,
            unet_config=config.MUSETALK_UNET_CONFIG,
            device=self.device,
        )
        self.vae, self.unet, self.pe = vae, unet, pe
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype
        self.timesteps = torch.tensor([0], device=self.device)

        self.audio_processor = mt.AudioProcessor(feature_extractor_path=config.MUSETALK_WHISPER)
        whisper = mt.WhisperModel.from_pretrained(config.MUSETALK_WHISPER)
        self.whisper = whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

    @torch.no_grad()
    def render(self, audio_path: str, output_path: str,
               with_audio: bool = True, frame_limit: int | None = None) -> str | None:
        """Render one clip. Returns None if the render was cut short by shutdown.

        Unlike FLOAT's ODE sampler (which ran to completion inside the external
        repo and could not be interrupted), the batch loop is here, so shutdown
        is checked every batch.
        """
        mt = _musetalk()
        m = self.material
        fps = config.MUSETALK_FPS

        features, librosa_length = self.audio_processor.get_audio_feature(
            audio_path, weight_dtype=self.weight_dtype)
        chunks = self.audio_processor.get_whisper_chunk(
            features, self.device, self.weight_dtype, self.whisper, librosa_length,
            fps=fps,
            audio_padding_length_left=config.MUSETALK_AUDIO_PAD_LEFT,
            audio_padding_length_right=config.MUSETALK_AUDIO_PAD_RIGHT,
        )

        frames_dir = tempfile.mkdtemp(prefix="mtframes_", dir=config.TEMP_DIR)
        idx = 0
        try:
            # device=: datagen's tail batch does a .to() with a cuda:0 default,
            # which would touch GPU 0 from every worker in a multi-GPU pool.
            for whisper_batch, latent_batch in mt.datagen(
                    chunks, m.latents, config.MUSETALK_BATCH_SIZE,
                    device=str(self.device)):
                if config.SHUTTING_DOWN.is_set():
                    return None
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                pred_latents = self.unet.model(
                    latent_batch, self.timesteps,
                    encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                for res_frame in self.vae.decode_latents(pred_latents):
                    if frame_limit is not None and idx >= frame_limit:
                        break
                    i = idx % len(m)
                    x1, y1, x2, y2 = m.coords[i]
                    try:
                        face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except cv2.error:
                        continue
                    # .copy(): the cached frame is reused by every later render,
                    # so blending must never write through to it.
                    combined = mt.get_image_blending(
                        m.frames[i].copy(), face, [x1, y1, x2, y2],
                        m.masks[i], m.mask_coords[i])
                    cv2.imwrite(os.path.join(frames_dir, f"{idx:08d}.png"), combined)
                    idx += 1
                if frame_limit is not None and idx >= frame_limit:
                    break

            if idx == 0:
                logger.warning("MuseTalk produced no frames for %s", audio_path)
                return None
            return _encode(frames_dir, output_path, audio_path if with_audio else None)
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)


class MuseTalkGPUPool:
    """Pool of MuseTalk models across multiple GPUs for parallel video generation."""

    def __init__(self, gpu_ids: list[int] = config.MUSETALK_GPUS):
        self.gpu_ids = gpu_ids
        self.workers = {}
        self.semaphores = {}
        self._lock = threading.Lock()

    def load_all(self):
        """Load MuseTalk on each GPU and attach the driving material. Call at startup."""
        material = None
        for gpu_id in self.gpu_ids:
            logger.info(f"Loading MuseTalk on GPU {gpu_id}...")
            worker = _Worker(gpu_id)
            with torch.cuda.device(gpu_id):
                worker.load()
                if material is None:
                    # Prepared once and shared: frames/masks are numpy and the
                    # latents live on the CPU until a batch is dispatched, so a
                    # second GPU costs no second copy.
                    material = _get_material(worker.vae)
            worker.material = material
            self.workers[gpu_id] = worker
            self.semaphores[gpu_id] = threading.Semaphore(1)
        logger.info(f"MuseTalk GPU pool ready: {list(self.workers.keys())}")

    def generate_video(self, audio_path: str, output_path: str | None = None) -> str | None:
        """Generate video using any available GPU from the pool."""
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4", dir=config.TEMP_DIR)

        # Try to acquire any GPU
        while True:
            # Bail out promptly on server shutdown so Ctrl+C isn't blocked waiting
            # here for a free GPU (the caller treats None as a failed render).
            if config.SHUTTING_DOWN.is_set():
                return None
            for gpu_id in self.gpu_ids:
                if self.semaphores[gpu_id].acquire(blocking=False):
                    try:
                        # Pin the current CUDA device for this thread so any
                        # device-less tensor creation inside MuseTalk lands on
                        # THIS gpu instead of the default cuda:0. Without this,
                        # segments dispatched to GPU 1/2 crash with "tensors on
                        # cuda:1 and cuda:0".
                        with torch.cuda.device(gpu_id):
                            return self.workers[gpu_id].render(audio_path, output_path)
                    finally:
                        self.semaphores[gpu_id].release()
            # All GPUs busy, wait briefly
            time.sleep(0.1)

    def generate_idle_video(self, duration: float = 5.0, output_path: str | None = None) -> str:
        """Generate a seamless moving idle clip with a stable smiling mouth."""
        if output_path is None:
            output_path = config.IDLE_VIDEO_PATH
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return _generate_motion_idle(duration, output_path)


def get_pool() -> MuseTalkGPUPool:
    global _pool
    # Double-checked locking: build the pool FULLY (load_all) before publishing it
    # to `_pool`. Otherwise concurrent callers during the cold load would see a
    # half-built pool with empty `workers`/`semaphores` -> KeyError on gpu_id.
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = MuseTalkGPUPool()
                pool.load_all()
                _pool = pool
    return _pool


def generate_video(audio_path: str, output_path: str | None = None) -> str | None:
    """Public API - uses the GPU pool."""
    return get_pool().generate_video(audio_path, output_path)


def generate_idle_video(duration: float = 3.0) -> str:
    """Generate a moving idle loop, preparing MuseTalk material if necessary."""
    os.makedirs(os.path.dirname(config.IDLE_VIDEO_PATH), exist_ok=True)
    if not os.path.exists(os.path.join(_material_dir(), "info.json")):
        return get_pool().generate_idle_video(duration)
    return _generate_motion_idle(duration, config.IDLE_VIDEO_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    pool = get_pool()
    print("MuseTalk GPU pool loaded successfully.")
