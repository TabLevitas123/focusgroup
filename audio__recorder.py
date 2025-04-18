import queue, threading, sounddevice as sd, soundfile as sf, time
from pathlib import Path
from utils.logger import get_logger
from utils.config import CFG

LOG = get_logger("Recorder")

class Recorder(threading.Thread):
    def __init__(self, outfile: Path):
        super().__init__(daemon=True)
        self.outfile = outfile
        self._q = queue.Queue(maxsize=100)
        self._stop = threading.Event()

    def callback(self, indata, frames, time_, status):
        if status:
            LOG.warning("Sounddevice status: %s", status)
        try:
            self._q.put_nowait(indata.copy())
        except queue.Full:
            LOG.warning("Audio queue full. Dropping frame.")

    def run(self):
        samplerate = CFG.get("SAMPLE_RATE", 16000)
        if not isinstance(samplerate, int):
            raise TypeError(f"Invalid sample rate: {samplerate}")
        try:
            with sf.SoundFile(self.outfile, mode="w", samplerate=samplerate, channels=1, subtype="PCM_16") as f:
                with sd.InputStream(samplerate=samplerate, channels=1, callback=self.callback):
                    LOG.info("Recording started")
                    while not self._stop.is_set():
                        try:
                            data = self._q.get(timeout=0.1)
                            f.write(data)
                        except queue.Empty:
                            continue
        except Exception as e:
            LOG.error("Recording failed: %s", e)
        LOG.info("Recording stopped.")

    def stop(self):
        self._stop.set()
        self.join(timeout=5)

def record_session(dest: Path, duration: float | None = None):
    rec = Recorder(dest)
    rec.start()
    if duration:
        time.sleep(duration)
        rec.stop()
    return rec