import sys, threading, tempfile, json, datetime
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QListWidget, QTextEdit, QProgressBar
)
from utils.logger import get_logger
from audio.recorder import record_session
from processing import transcriber, diarizer, profile_builder, recommender
from storage import database

LOG = get_logger("GUI")

class FocusPanelApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusPanel Analyzer")
        self.recorder = None
        self.wav_path: Path | None = None
        self.profiles = []
        self.recs = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.start_btn = QPushButton("Start Conversation")
        self.stop_btn = QPushButton("End Conversation")
        self.compile_btn = QPushButton("Compile & Analyze")
        self.export_btn = QPushButton("Export Session")
        self.status = QLabel("Idle.")
        self.list_profiles = QListWidget()
        self.details = QTextEdit(readOnly=True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        for b in (self.stop_btn, self.compile_btn):
            b.setEnabled(False)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.compile_btn.clicked.connect(self._compile)
        self.export_btn.clicked.connect(self._export)
        self.list_profiles.itemSelectionChanged.connect(self._show_profile)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.compile_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.list_profiles)
        layout.addWidget(self.details)

    def _start(self):
        self.wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        self.recorder = record_session(self.wav_path)
        self.status.setText("Recording...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        if self.recorder:
            # Now we can directly call stop() on our RecordingThread
            self.recorder.stop()
            self.recorder = None
        self.status.setText(f"Saved to {self.wav_path}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.compile_btn.setEnabled(True)

    def _compile(self):
        self.status.setText("Transcribing…")
        threading.Thread(target=self._pipeline, daemon=True).start()

    def _pipeline(self):
        self.progress.setValue(10)
        whisper_json = transcriber.transcribe_wav(self.wav_path)
        self.progress.setValue(40)
        diar = diarizer.diarize(self.wav_path)
        self.progress.setValue(60)
        segments = transcriber.extract_segments(whisper_json)
        self.progress.setValue(80)

        utterances = []
        for turn in diar.itertracks(yield_label=True):
            seg, _, speaker = turn
            text = next((s["text"] for s in segments if abs(s["start"] - seg.start) < 0.1), "")
            utterances.append(profile_builder.Utterance(seg.start, seg.end, speaker, text))

        self.profiles = profile_builder.build_profiles(utterances)
        self.recs = recommender.recommend(self.profiles)
        database.save_session(str(self.wav_path), profile_builder.profiles_to_json(self.profiles), self.recs)

        self.list_profiles.clear()
        for p in self.profiles:
            self.list_profiles.addItem(p.speaker)
        self.status.setText("Analysis complete. Select a speaker.")
        self.details.setPlainText("Recommendations:\n" + json.dumps(self.recs, indent=2))
        self.progress.setValue(100)

    def _show_profile(self):
        idx = self.list_profiles.currentRow()
        if idx < 0: return
        p = self.profiles[idx]
        self.details.setPlainText(json.dumps(profile_builder.asdict(p), indent=2))

    def _export(self, export_dir=None):
        """Export session data to a JSON file.
        
        Args:
            export_dir: Optional directory path where to save the export.
                        If None, uses the home directory.
        """
        base_dir = Path(export_dir) if export_dir else Path.home()
        out = base_dir / f"focuspanel_export_{datetime.datetime.now().isoformat().replace(':', '-')}.json"
        with open(out, "w") as f:
            json.dump({
                "profiles": profile_builder.profiles_to_json(self.profiles),
                "recommendations": self.recs
            }, f, indent=2)
        self.status.setText(f"Exported session to {out}")
        return out  # Return the path for testing

def run():
    database.init()
    app = QApplication(sys.argv)
    w = FocusPanelApp()
    w.resize(600, 800)
    w.show()
    sys.exit(app.exec())