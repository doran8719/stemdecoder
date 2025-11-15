import sys
import subprocess
from pathlib import Path

def main():
    print("=== AI Stem Splitter (V1) ===")

    # 1) If a file path is passed as an argument, use it
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # 2) Otherwise, ask the user and let them drag+drop
        audio_path = input("Drag your audio file here and press Enter: ").strip()

    if not audio_path:
        print("No file path provided. Exiting.")
        return

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"File not found: {audio_file}")
        return

    print(f"\nRunning Demucs on: {audio_file}")
    try:
        subprocess.run(["demucs", str(audio_file)], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running Demucs:", e)
        return

    print("\nDone! Stems should be in the 'separated' folder in this directory.")
    print("You can open it in Finder and drag the stems into Ableton/your DAW.")

if __name__ == "__main__":
    main()

