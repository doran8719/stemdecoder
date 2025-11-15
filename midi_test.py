from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
from pathlib import Path

print("Drag an audio file (preferably a single stem) and press Enter:")
audio_path = input().strip()
audio_file = Path(audio_path)

if not audio_file.exists():
    print("File not found:", audio_file)
    exit()

output_dir = Path("midi_output")
output_dir.mkdir(exist_ok=True)

print("\nRunning Basic Pitch model...")

# Basic Pitch expects:
# 1) list of input paths
# 2) output directory
# 3) save_midi (bool)
# 4) sonify_midi (bool)
# 5) save_model_outputs (bool)
# 6) save_notes (bool)
# 7) model_path (usually ICASSP_2022_MODEL_PATH)
predict_and_save(
    [str(audio_file)],           # input-audio-path-list
    str(output_dir),             # output-directory
    True,                        # save_midi
    False,                       # sonify_midi (render MIDI to WAV)
    False,                       # save_model_outputs (NPZ)
    False,                       # save_notes (CSV)
    ICASSP_2022_MODEL_PATH       # model path
)

print("\nDone! Check the midi_output folder for your .mid file.")

