# workaround for sounddevice and soundfile: https://github.com/ohmtech-rdi/eurorack-blocks/issues/444
import sys
import sounddevice as sd
import soundfile as sf

def play_audio(file_path):
    try:
        data, fs = sf.read(file_path)
        sd.play(data, fs)
        sd.wait()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audio_file_path = sys.argv[1]
    play_audio(audio_file_path)
