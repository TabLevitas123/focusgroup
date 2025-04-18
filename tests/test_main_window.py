import pytest
import json
from gui import main_window
from PySide6.QtWidgets import QApplication

# Mark all tests in this file as requiring GUI
pytestmark = pytest.mark.skip(reason="GUI tests require OpenGL libraries not available in this environment")

@pytest.fixture(scope="module")
def app():
    return QApplication([])

def test_gui_start_stop_cycle(qtbot, app):
    window = main_window.FocusPanelApp()
    qtbot.addWidget(window)

    window._start()
    assert not window.start_btn.isEnabled()
    assert window.stop_btn.isEnabled()

    window._stop()
    assert window.start_btn.isEnabled()
    assert not window.stop_btn.isEnabled()

def test_gui_export_format(tmp_path, qtbot, app):
    window = main_window.FocusPanelApp()
    qtbot.addWidget(window)

    window.profiles = []
    window.recs = {"tone": "neutral"}
    export_path = window._export(export_dir=tmp_path)
    
    # Check that the file was created
    assert export_path.exists()
    
    # Check file content
    with open(export_path, 'r') as f:
        data = json.loads(f.read())
        assert "profiles" in data
        assert "recommendations" in data
        assert data["recommendations"]["tone"] == "neutral"