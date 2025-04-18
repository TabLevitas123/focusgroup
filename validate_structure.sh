#!/bin/bash
echo "🧪 Validating FocusPanel structure..."
required_paths=(
    "utils/config.py"
    "utils/logger.py"
    "audio/recorder.py"
    "processing/transcriber.py"
    "processing/diarizer.py"
    "processing/profile_builder.py"
    "processing/recommender.py"
    "gui/main_window.py"
    "storage/database.py"
    "tests/test_recorder.py"
    "tests/test_config.py"
    "tests/test_logger.py"
    "tests/test_transcriber.py"
    "tests/test_profile_builder.py"
    "tests/test_recommender.py"
    "tests/test_database.py"
    "tests/test_diarizer.py"
    "tests/test_main_window.py"
)
missing=0
for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "❌ Missing: $path"
    missing=1
  else
    echo "✅ Found:   $path"
  fi
done
exit $missing