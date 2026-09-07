/* Microphone capture for the counselor's audio path.
 *
 * There is deliberately no DSP here. Echo cancellation (WebRTC AEC3), noise
 * suppression and gain control all run inside the browser's own audio pipeline,
 * ahead of this node — see the getUserMedia constraints in index.html. Doing any
 * of it again on this side would only fight them: the browser's canceller is
 * non-linear, and a second stage cannot undo what it has already reshaped.
 *
 * All this node does is batch the stream and convert it to the int16 the audio
 * WebSocket carries. It replaces a ScriptProcessorNode, which is deprecated and
 * ran the same conversion on the main thread, where it competed with video
 * playback and the chat UI.
 */
'use strict';

// One Silero VAD chunk (modules/vad.py splits incoming audio on 512 samples, so
// anything that is not a multiple of 512 gets zero-padded and corrupts the tail).
// 512 at 16 kHz = 32 ms per message, which also keeps ASR-confirmed barge-in
// (BARGE_IN_RECHECK in config.py) responsive.
const CHUNK = 512;

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;                 // mic not connected yet, or a silent render quantum
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === CHUNK) {
        const pcm = new Int16Array(CHUNK);
        for (let k = 0; k < CHUNK; k++) {
          const c = this.buf[k] < -1 ? -1 : (this.buf[k] > 1 ? 1 : this.buf[k]);
          pcm[k] = c < 0 ? c * 0x8000 : c * 0x7fff;
        }
        this.port.postMessage(pcm, [pcm.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCapture);
