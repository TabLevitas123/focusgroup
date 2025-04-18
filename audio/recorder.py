import threading
import pyaudio
import wave
from utils.logger import get_logger

LOG = get_logger("Recorder")

class RecordingThread(threading.Thread):
    def __init__(self, output_path, channels=1, rate=16000, chunk=1024):
        super().__init__()
        self.output_path = output_path
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.daemon = True
        self.stop_event = threading.Event()
        self.frames = []
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        LOG.info(f"Recording initialized, will save to {output_path}")
        
    def run(self):
        try:
            LOG.info("Recording started...")
            while not self.stop_event.is_set():
                data = self.stream.read(self.chunk)
                self.frames.append(data)
        except Exception as e:
            LOG.error(f"Error while recording: {e}")
        finally:
            # Close everything
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            
            # Save the audio file
            with wave.open(self.output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
            LOG.info(f"Recording saved to {self.output_path}")
    
    def stop(self):
        """Stop the recording"""
        LOG.info("Stopping recording...")
        self.stop_event.set()

def record_session(output_path, duration=None, channels=1, rate=16000, chunk=1024):
    """
    Record audio to a WAV file
    
    Args:
        output_path: Path to save the WAV file
        duration: Recording duration in seconds (None for unlimited)
        channels: Number of audio channels
        rate: Sample rate
        chunk: Frames per buffer
        
    Returns:
        RecordingThread: A thread object with a stop() method to stop recording
    """
    # Create and start the recording thread
    thread = RecordingThread(output_path, channels, rate, chunk)
    thread.start()
    
    # If duration is specified, schedule the stop event
    if duration is not None:
        def stop_after_duration():
            import time
            time.sleep(duration)
            thread.stop()
        
        duration_thread = threading.Thread(target=stop_after_duration)
        duration_thread.daemon = True
        duration_thread.start()
    
    return thread