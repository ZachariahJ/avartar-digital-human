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

sys.path.insert(0, config.MUSETALK_DIR)

import torch

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()

_mt = None
_mt_lock = threading.Lock()

_NO_FACE = (0.0, 0.0, 0.0, 0.0)

_DONE = object()


@contextlib.contextmanager
def _cwd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _musetalk():
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
    if abs(src_fps - config.MUSETALK_FPS) > 0.5:
        logger.info("driving video is %.2f fps, rendering at %d fps — head motion "
                    "will play at %.2fx during speech; re-encode the clip to %d fps "
                    "to match the idle loop",
                    src_fps, config.MUSETALK_FPS,
                    config.MUSETALK_FPS / src_fps if src_fps else 0,
                    config.MUSETALK_FPS)
    return frames


def _prepare_material(mat_dir: str, vae) -> _Material:
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

    src_frames = _video_to_frames(config.AVATAR_VIDEO)
    for i, frame in enumerate(src_frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
    img_list = sorted(glob.glob(os.path.join(imgs_dir, "*.png")))

    coord_list, frame_list = mt.get_landmark_and_bbox(img_list, config.MUSETALK_BBOX_SHIFT)

    coords, frames, latents = [], [], []
    for bbox, frame in zip(coord_list, frame_list):
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
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(imgs_dir, f"{i:08d}.png"), frame)
        mask, crop_box = mt.get_image_prepare_material(frame, coords[i], fp=fp, mode=mode)
        cv2.imwrite(os.path.join(masks_dir, f"{i:08d}.png"), mask)
        masks.append(mask)
        mask_coords.append(crop_box)
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


class _Worker:

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.material = None

    def load(self):
        mt = _musetalk()
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
        mt = _musetalk()
        m = self.material
        fps = config.MUSETALK_FPS
        jpeg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), config.MUSETALK_JPEG_QUALITY]

        halt = threading.Event()
        done = threading.Event()
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
                            return
                        continue
                    if item is _DONE or _stop():
                        return
                    idx, res_frame = item
                    i = idx % len(m)
                    x1, y1, x2, y2 = m.coords[i]
                    face = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    combined = m.frames[i].copy()
                    blend.paste_face(combined, face, (x1, y1, x2, y2),
                                     m.masks[i], m.mask_coords[i])
                    ok, buf = cv2.imencode(".jpg", combined, jpeg_opts)
                    if not ok:
                        raise RuntimeError(f"JPEG encode failed on frame {idx}")
                    on_frame(state.emitted, buf.tobytes())
                    state.emitted += 1
            except BaseException as exc:
                state.error = exc
            finally:
                halt.set()

        def _offer(item) -> bool:
            while not _stop():
                try:
                    q.put(item, timeout=0.05)
                    state.high_water = max(state.high_water, q.qsize())
                    return True
                except queue.Full:
                    continue
            return False

        features, librosa_length = self._mel_features(audio_path)
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
            with contextlib.suppress(queue.Full):
                q.put_nowait(_DONE)
            worker.join()

        if state.error is not None:
            raise state.error
        if state.emitted == 0:
            logger.warning("MuseTalk produced no frames for %s", audio_path)
        return state.emitted


class MuseTalkGPUPool:

    def __init__(self, gpu_ids: list[int] = config.MUSETALK_GPUS):
        self.gpu_ids = gpu_ids
        self.workers = {}
        self.semaphores = {}
        self._lock = threading.Lock()

    def load_all(self):
        material = None
        for gpu_id in self.gpu_ids:
            logger.info(f"Loading MuseTalk on GPU {gpu_id}...")
            worker = _Worker(gpu_id)
            with torch.cuda.device(gpu_id):
                worker.load()
                if material is None:
                    material = _get_material(worker.vae)
            worker.material = material
            self.workers[gpu_id] = worker
            self.semaphores[gpu_id] = threading.Semaphore(1)
        logger.info(f"MuseTalk GPU pool ready: {list(self.workers.keys())}")

    def stream_video(self, audio_path: str, on_frame, abort=None) -> int:
        while True:
            if config.SHUTTING_DOWN.is_set() or (abort is not None and abort()):
                return 0
            for gpu_id in self.gpu_ids:
                if self.semaphores[gpu_id].acquire(blocking=False):
                    try:
                        with torch.cuda.device(gpu_id):
                            return self.workers[gpu_id].stream(audio_path, on_frame, abort)
                    finally:
                        self.semaphores[gpu_id].release()
            time.sleep(0.1)


def get_pool() -> MuseTalkGPUPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = MuseTalkGPUPool()
                pool.load_all()
                _pool = pool
    return _pool


def stream_video(audio_path: str, on_frame, abort=None) -> int:
    return get_pool().stream_video(audio_path, on_frame, abort)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pool = get_pool()
    print("MuseTalk GPU pool loaded successfully.")
