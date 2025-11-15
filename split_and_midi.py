import subprocess
from pathlib import Path
import shutil
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

print("=== AI Split + MIDI Extractor (V1) ===")
raw_path = input("Drag your full song here and press Enter: ").strip()

# Handle macOS drag-and-drop escaping (turn '\ ' back into real spaces)
audio_path = raw_path.replace("\\ ", " ")

# Also strip wrapping quotes if Terminal added any
if (audio_path.startswith('"') and audio_path.endswith('"')) or \
   (audio_path.startswith("'") and audio_path.endswith("'")):
    audio_path = audio_path[1:-1]

audio_file = Path(audio_path)

if not audio_file.exists():
    print(f"File not found: {audio_file}")
    exit()

# Create output folder
song_name = audio_file.stem
output_root = Path("output") / song_name
output_root.mkdir(parents=True, exist_ok=True)

print("\nRunning Demucs (stem separation)...")
try:
    subprocess.run(["demucs", str(audio_file)], check=True)
except subprocess.CalledProcessError as e:
    print("Error running Demucs:", e)
    exit()

# Find the Demucs output directory (usually separated/htdemucs/<song_name>/)
separated_dir = Path("separated")
stem_dir = None

for model_folder in separated_dir.iterdir():
    candidate = model_folder / song_name
    if candidate.exists():
        stem_dir = candidate
        break

if not stem_dir:
    print("Could not find Demucs output folder.")
    exit()

print(f"Demucs output found at: {stem_dir}")

# Copy stems to output folder
for stem_file in stem_dir.iterdir():
    if stem_file.suffix.lower() == ".wav":
        shutil.copy(stem_file, output_root / stem_file.name)

print("\nExtracting MIDI for melodic stems...")
melodic_stems = ["other.wav", "bass.wav"]

for stem_name in melodic_stems:
    stem_path = output_root / stem_name
    if stem_path.exists():
        output_midi_dir = output_root / "midi"
        output_midi_dir.mkdir(exist_ok=True)

        print(f"Running Basic Pitch on {stem_name}...")

        predict_and_save(
            [str(stem_path)],      # list of paths
            str(output_midi_dir),  # directory
            True,                  # save MIDI
            False,                 # don't sonify MIDI
            False,                 # don't save model outputs
            False,                 # don't save notes CSV
            ICASSP_2022_MODEL_PATH
        )

print("\n=== DONE ===")
print(f"All results saved to: {output_root.resolve()}")

