"""Renders the talking head, one frame at a time as it is produced.

Public API: get_pool() and stream_video().

Nothing here produces a video file. Each blended frame is handed to a callback
as JPEG bytes the moment the VAE decoder returns it, and the browser draws those
on a canvas against separately fetched audio. Since the UNet works in batches,
the first frames are available after one batch rather than after the whole
utterance — which is the reason for the arrangement.

MuseTalk repaints only the mouth region of an existing clip, so head motion and
every pixel outside the mouth come from config.AVATAR_VIDEO. Deriving that
clip's face boxes, latents and blend masks costs minutes, so it is done once per
avatar and reused by every render afterwards.
"""
import contextlib
import glob
import json
import logging
import os
import pickle
import queue
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import librosa
import numpy as np

import config
from modules import blend

# MuseTalk is a sibling checkout rather than an installed package, and the
# import below reaches into it, so this must precede it.
sys.path.insert(0, config.MUSETALK_DIR)

import torch

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()

_mt = None          # handles to MuseTalk's own functions, imported once
_mt_lock = threading.Lock()

# What MuseTalk returns as a bounding box when it found no face.
_NO_FACE = (0.0, 0.0, 0.0, 0.0)

# Queued after the last frame so the post-processing thread knows to stop.
_DONE = object()


@contextlib.contextmanager
def _cwd(path: str):
    """Temporarily change the process working directory. See _musetalk()."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _musetalk():
    """Import MuseTalk's modules once, from inside its own directory.

    The directory change is required, not tidiness. Upstream builds its DWPose
    landmark model at import time from paths relative to the repository root,
    and its face parser resolves its checkpoint the same way; imported from
    anywhere else, both fail to find their weights.

    Since the working directory is process-global, it is confined to this import
    and to material preparation, both of which run under the pool lock during
    startup, before any request is served.
    """
    global _mt
    if _mt is None:
        with _mt_lock:
            if _mt is None:
                t0 = time.perf_counter()
                with _cwd(config.MUSETALK_DIR):
                    from musetalk.utils.utils import load_all_model, datagen
                    from musetalk.utils.preprocessing import get_landmark_and_bbox
                    from musetalk.utils.blending import get_image_prepare_material
                    from musetalk.utils.face_parsing import FaceParsing
                    from musetalk.utils.audio_processor import AudioProcessor
                    from transformers import WhisperModel
                _mt = SimpleNamespace(
                    load_all_model=load_all_model,
                    datagen=datagen,
                    get_landmark_and_bbox=get_landmark_and_bbox,
                    get_image_prepare_material=get_image_prepare_material,
                    FaceParsing=FaceParsing,
                    AudioProcessor=AudioProcessor,
                    WhisperModel=WhisperModel,
                )
                logger.info("MuseTalk imported in %.1fs", time.perf_counter() - t0)
    return _mt


class _Material:
    """One prepared cycle of driving frames, ready to render against.

    The cycle is the source clip forwards then backwards, so an utterance longer
    than the clip wraps without a visible jump — the last frame is adjacent to
    the first. All five lists are aligned: index i describes the same frame in
    every one of them.

    Frames are held decoded in memory, roughly three bytes per pixel for twice
    the clip's length, which is why the driving video should stay short.
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
    """Decode the driving video to BGR frames, warning on a frame-rate mismatch."""
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
    # Not fatal, and lip sync is unaffected — audio sampling and browser
    # playback both index by MUSETALK_FPS, so they agree whatever the clip was
    # shot at. What breaks is head-motion speed: one driving frame is consumed
    # per output frame, so the clip plays at the wrong rate during speech and
    # visibly changes pace against the natively-played idle loop.
    if abs(src_fps - config.MUSETALK_FPS) > 0.5:
        logger.info("driving video is %.2f fps, rendering at %d fps — head motion "
                    "will play at %.2fx during speech; re-encode the clip to %d fps "
                    "to match the idle loop",
                    src_fps, config.MUSETALK_FPS,
                    config.MUSETALK_FPS / src_fps if src_fps else 0,
                    config.MUSETALK_FPS)
    return frames


def _prepare_material(mat_dir: str, vae) -> _Material:
    """Derive face boxes, latents and blend masks from the driving video.

    Takes minutes, so it runs once per avatar. Everything is built in a scratch
    directory and renamed into place at the end: an interrupted run must not
    leave a half-built directory that a later startup would happily load.
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

    # Upstream's landmark detector takes file paths, not arrays, so the decoded
    # frames have to reach disk before it can see them.
    src_frames = _video_to_frames(config.AVATAR_VIDEO)
    for i, frame in enumerate(src_frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
    img_list = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))

    coord_list, frame_list = mt.get_landmark_and_bbox(img_list, config.MUSETALK_BBOX_SHIFT)

    coords, frames, latents = [], [], []
    for bbox, frame in zip(coord_list, frame_list):
        # Drop the whole frame, not just its latent. Skipping only the latent —
        # as upstream does — shifts every subsequent latent onto the wrong
        # image, so the lists must be kept the same length here.
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

    # Appending the reverse makes the cycle wrap without a jump cut.
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
    # Rewritten in cycle order, so that a later reload reconstructs exactly this
    # sequence rather than the raw decode.
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
        mask, crop_box = mt.get_image_prepare_material(frame, coords[i], fp=fp, mode=mode)
        cv2.imwrite(os.path.join(masks_dir, f"{i:08d}.png"), mask)
        masks.append(mask)
        mask_coords.append(crop_box)
    # Dropped frames leave surplus files behind, and the reload globs the
    # directory, so it would pick them back up.
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
    """Reload material prepared by an earlier run.

    Raises if the five lists disagree in length: a truncated directory would
    otherwise render frames against the wrong masks and coordinates.
    """
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
    """Material for the current avatar, preparing it if this is the first run."""
    mat_dir = _material_dir()
    if os.path.exists(os.path.join(mat_dir, "latents.pt")):
        return _load_material(mat_dir)
    os.makedirs(config.MUSETALK_MATERIAL_DIR, exist_ok=True)
    return _prepare_material(mat_dir, vae)


class _Worker:
    """One complete MuseTalk model stack, pinned to one GPU."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.material = None

    def load(self):
        """Load the models onto this worker's GPU, in half precision."""
        mt = _musetalk()
        # Upstream builds the VAE path as os.path.join("models", vae_type).
        # Passing an absolute vae_type makes that join return it unchanged,
        # which locates the weights without needing another directory change.
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

    def _mel_features(self, audio_path: str):
        """Whisper mel features for one utterance, computed on this worker's GPU.

        Upstream's AudioProcessor.get_audio_feature runs the STFT through
        WhisperFeatureExtractor's numpy path: a Python loop over frames costing
        ~380ms whatever the utterance's length, because every segment is padded
        out to whisper's full 30s window. That was the whole time-to-first-frame,
        and it fell hardest on the short replies a conversation is mostly made
        of. The same extractor has a torch path behind its `device` argument —
        same filterbank, ~26ms. The two differ by 3e-05, and the result is cast
        to fp16 here, whose resolution is 30x coarser than that.
        """
        speech, sr = librosa.load(audio_path, sr=16000)
        window = 30 * sr
        extract = self.audio_processor.feature_extractor
        features = [
            extract(speech[i:i + window], return_tensors="pt", sampling_rate=sr,
                    device=str(self.device)).input_features.to(dtype=self.weight_dtype)
            for i in range(0, len(speech), window)
        ]
        return features, len(speech)

    @torch.no_grad()
    def stream(self, audio_path: str, on_frame, abort=None) -> int:
        """Render one utterance, emitting each frame as soon as it is blended.

        Args:
            audio_path: the utterance's audio. A path, not bytes, because the
                whisper feature extractor takes a filename.
            on_frame: called as on_frame(index, jpeg_bytes) once per frame, in
                order, while the render is still running. Invoked from the
                post-processing thread, never from the caller's.
            abort: polled per batch and per frame, from both threads, so it
                must be thread-safe; when it returns true the render stops
                where it is.

        Returns:
            How many frames were emitted, which is short of the full count if
            the render was aborted.

        Blending and JPEG encoding run on a second thread rather than inline.
        They cost ~13ms a frame against the GPU's ~33ms, so inline they simply
        added up — 200ms of UNet then 76ms of idle GPU, per batch of six, which
        is what held the renderer below the 24fps the browser plays at. Both
        stages now run at once and the GPU alone sets the pace. cv2 releases
        the GIL, so the two threads genuinely overlap.
        """
        mt = _musetalk()
        m = self.material
        fps = config.MUSETALK_FPS
        jpeg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), config.MUSETALK_JPEG_QUALITY]

        halt = threading.Event()    # 消费端出错，生产端停手
        done = threading.Event()    # 生产端结束，消费端可退
        # Holds the 256x256 decoded faces, not the 1.5MB composited frames, so
        # two batches in flight is a couple of megabytes. Sized for jitter, not
        # throughput: the consumer outruns the GPU, so it is normally empty.
        q = queue.Queue(maxsize=2 * config.MUSETALK_BATCH_SIZE)
        state = SimpleNamespace(emitted=0, error=None, high_water=0)

        def _stop():
            return (halt.is_set() or config.SHUTTING_DOWN.is_set()
                    or (abort is not None and abort()))

        def _consume():
            try:
                while True:
                    try:
                        item = q.get(timeout=0.05)
                    except queue.Empty:
                        if done.is_set():
                            return              # 生产端已死，不再等
                        continue
                    if item is _DONE or _stop():
                        return
                    idx, res_frame = item
                    i = idx % len(m)
                    x1, y1, x2, y2 = m.coords[i]
                    face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    # Copy first: this frame is shared with every future render,
                    # so blending in place would permanently deface the material.
                    combined = m.frames[i].copy()
                    blend.paste_face(combined, face, (x1, y1, x2, y2),
                                     m.masks[i], m.mask_coords[i])
                    ok, buf = cv2.imencode(".jpg", combined, jpeg_opts)
                    # 静默丢帧会让浏览器索引持续错位
                    if not ok:
                        raise RuntimeError(f"JPEG encode failed on frame {idx}")
                    on_frame(state.emitted, buf.tobytes())
                    state.emitted += 1
            except BaseException as exc:
                state.error = exc
            finally:
                halt.set()          # 生产端别再往满队列里塞

        def _offer(item) -> bool:
            """Hand one frame to the consumer; false means stop rendering."""
            while not _stop():
                try:
                    q.put(item, timeout=0.05)
                    state.high_water = max(state.high_water, q.qsize())
                    return True
                except queue.Full:
                    continue
            return False

        features, librosa_length = self._mel_features(audio_path)
        # The audio cannot be streamed in, only the output out: chunking trims
        # and pads against the total length, so the whole utterance must exist
        # before any of it can be processed.
        chunks = self.audio_processor.get_whisper_chunk(
            features, self.device, self.weight_dtype, self.whisper, librosa_length,
            fps=fps,
            audio_padding_length_left=config.MUSETALK_AUDIO_PAD_LEFT,
            audio_padding_length_right=config.MUSETALK_AUDIO_PAD_RIGHT,
        )

        worker = threading.Thread(target=_consume, name="musetalk-post", daemon=True)
        worker.start()
        idx = 0
        try:
            # Passing device explicitly: upstream's final partial batch calls
            # .to() with a cuda:0 default, so without this every worker in a
            # multi-GPU pool reaches onto GPU 0.
            for whisper_batch, latent_batch in mt.datagen(
                    chunks, m.latents, config.MUSETALK_BATCH_SIZE,
                    device=str(self.device)):
                if _stop():
                    break
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                pred_latents = self.unet.model(
                    latent_batch, self.timesteps,
                    encoder_hidden_states=audio_feature_batch).sample
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                for res_frame in self.vae.decode_latents(pred_latents):
                    if not _offer((idx, res_frame)):
                        break
                    idx += 1
                else:
                    continue
                break
        finally:
            done.set()
            # 消费端可能已死，队列满时塞不进去
            with contextlib.suppress(queue.Full):
                q.put_nowait(_DONE)
            worker.join()

        # A producer exception has already propagated past here; this is the
        # consumer's, re-raised on the caller's thread so the first failure wins.
        if state.error is not None:
            raise state.error
        if state.emitted == 0:
            logger.warning("MuseTalk produced no frames for %s", audio_path)
        return state.emitted


class MuseTalkGPUPool:
    """One loaded model stack per GPU, so utterances can render in parallel."""

    def __init__(self, gpu_ids: list[int] = config.MUSETALK_GPUS):
        self.gpu_ids = gpu_ids
        self.workers = {}
        self.semaphores = {}
        self._lock = threading.Lock()

    def load_all(self):
        """Load a worker on every configured GPU. Slow; call once at startup."""
        material = None
        for gpu_id in self.gpu_ids:
            logger.info(f"Loading MuseTalk on GPU {gpu_id}...")
            worker = _Worker(gpu_id)
            with torch.cuda.device(gpu_id):
                worker.load()
                if material is None:
                    # Prepared once and shared across workers. Frames and masks
                    # are numpy and the latents stay on the CPU until a batch is
                    # dispatched, so extra GPUs cost no extra copies.
                    material = _get_material(worker.vae)
            worker.material = material
            self.workers[gpu_id] = worker
            self.semaphores[gpu_id] = threading.Semaphore(1)
        logger.info(f"MuseTalk GPU pool ready: {list(self.workers.keys())}")

    def stream_video(self, audio_path: str, on_frame, abort=None) -> int:
        """Render one utterance on whichever GPU frees up first.

        Blocks the calling thread for the entire render and invokes `on_frame`
        from it, so callers must run this off the event loop and `on_frame` must
        not block — it should queue the frame and return.
        """
        while True:
            # Aborting is checked while waiting for a GPU, not only once one is
            # held. An interrupted utterance must not queue for a GPU it will
            # never use, and the pre-warm — whose abort condition is "somebody
            # is talking" — must not grab the first GPU a live conversation
            # releases.
            if config.SHUTTING_DOWN.is_set() or (abort is not None and abort()):
                return 0
            for gpu_id in self.gpu_ids:
                if self.semaphores[gpu_id].acquire(blocking=False):
                    try:
                        # Pins this thread's CUDA device so that any tensor
                        # MuseTalk creates without an explicit device lands
                        # here. Without it, work dispatched to any GPU but the
                        # first fails with tensors split across two devices.
                        with torch.cuda.device(gpu_id):
                            return self.workers[gpu_id].stream(audio_path, on_frame, abort)
                    finally:
                        self.semaphores[gpu_id].release()
            time.sleep(0.1)


def get_pool() -> MuseTalkGPUPool:
    """The GPU pool, loaded on first use.

    The pool is fully loaded before being published to the global. Assigning it
    first would let a concurrent caller find it with empty worker and semaphore
    maps and fail on a missing GPU id.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = MuseTalkGPUPool()
                pool.load_all()
                _pool = pool
    return _pool


def stream_video(audio_path: str, on_frame, abort=None) -> int:
    """Render one utterance on the shared pool. See MuseTalkGPUPool.stream_video."""
    return get_pool().stream_video(audio_path, on_frame, abort)


if __name__ == "__main__":
    # Smoke test: model loading and material preparation are what break here,
    # and both happen before this prints.
    logging.basicConfig(level=logging.INFO)
    pool = get_pool()
    print("MuseTalk GPU pool loaded successfully.")
