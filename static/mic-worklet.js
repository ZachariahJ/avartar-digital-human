/* Batches microphone audio and converts it to the format the server expects.
 *
 * That is all it does, and the absence of any signal processing is deliberate.
 * Echo cancellation, noise suppression and gain control already run inside the
 * browser ahead of this node — see the getUserMedia constraints in index.html.
 * Adding a second stage here would fight them rather than help: the browser's
 * canceller is non-linear, so nothing downstream can undo what it has done.
 *
 * Runs on the audio thread. Doing the same work on the main thread makes it
 * compete with video playback and the chat UI, which is audible.
 */
'use strict';

// Exactly one chunk of what the server's voice detector consumes. Sending any
// other size means the remainder is zero-padded on that side, which corrupts
// the tail of every message. At 16 kHz this is 32ms per message, which is also
// what keeps interrupting the avatar feel immediate.
const CHUNK = 512;

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;                 // mic not connected, or a silent quantum
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === CHUNK) {
        const pcm = new Int16Array(CHUNK);
        for (let k = 0; k < CHUNK; k++) {
          // Clamp before scaling: a sample outside [-1, 1] wraps around on
          // conversion, turning a loud passage into a burst of noise. The two
          // scale factors differ because the int16 range is asymmetric.
          const c = this.buf[k] < -1 ? -1 : (this.buf[k] > 1 ? 1 : this.buf[k]);
          pcm[k] = c < 0 ? c * 0x8000 : c * 0x7fff;
        }
        // Transferred rather than copied, so each message costs no allocation
        // on the receiving side.
        this.port.postMessage(pcm, [pcm.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCapture);
