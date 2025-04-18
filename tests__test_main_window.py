import pytest
from gui import main_window
from PySide6.QtWidgets import QApplication

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

def test_gui_export_format(tmp_path, qtbot):
    window = main_window.FocusPanelApp()
    qtbot.addWidget(window)

    window.profiles = []
    window.recs = {"tone": "neutral"}
    window._export()