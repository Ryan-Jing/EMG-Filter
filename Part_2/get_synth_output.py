import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import wave

# BASE PARAMETERS
CHUNK_LENGTH = 160*4 # 160 is 10ms because sample rate is 16kHz
OVERLAP = 0
CENTRE_FREQUENCY = 100
SAMPLING_RATE = 16e3
TYPE = "butterworth"
BAND_WIDTH = 100
ORDER = 3
NUMBER_OF_FILTERS = 150
INTERVAL = 50
TITLE = f"./syth_voice_chunk_length_{CHUNK_LENGTH}"
OUTPUT_FILE = f"{TITLE}.wav"
OUTPUT_GRAPH = f"{TITLE}.png"
OUTPUT_RMS_VALS = f"{TITLE}_rms_vals.png"
OUTPUT_SIN_VALS = f"{TITLE}_sin_vals.png"


rms_values = []
sin_values = []

def create_bandpass_filters(centre_frequency, band_width, filter_type, number_of_filters, order, interval, sample_rate=SAMPLING_RATE):
    filters = [create_bandpass_filter(
            filter_type=filter_type, 
            order=order, 
            centre_freq=centre_frequency + interval*i, 
            band_width=band_width, 
            sample_rate=sample_rate) 
        for i in range(number_of_filters)
    ]
    return filters

def create_bandpass_filter(filter_type: str, order: int, centre_freq: int, band_width: int, sample_rate: int):
    lower_freq = centre_freq - band_width / 2
    upper_freq = centre_freq + band_width / 2
    
    if filter_type == "butterworth":
        return signal.butter(
            order, 
            [lower_freq, upper_freq], 
            btype='bandpass', 
            output='sos', 
            fs=sample_rate
        )
    elif filter_type == "chebyshev":
        return signal.cheby2(
            order, 
            20, 
            [lower_freq, upper_freq], 
            btype='bandpass', 
            output='sos', 
            fs=sample_rate
        )
    else:
        raise ValueError("Filter type must be either 'butterworth' or 'chebyshev'")

def filter_chunk(filter, chunk):
    return signal.sosfilt(filter, chunk)

def get_rms(filtered_chunk):
    rms = np.sqrt(np.mean(filtered_chunk**2))
    rms_values.append(rms)
    return rms

def synthesize(rms, chunk_length, centre_freq_band, start_time, sampling_rate=SAMPLING_RATE):
    # time = np.arange(start_time, start_time+chunk_length, chunk_length/SAMPLING_RATE)
    time = np.arange(start_time, start_time+chunk_length)
    time_seconds = time / sampling_rate
    sin_wave = np.sin(2 * np.pi * centre_freq_band * time_seconds)
    # t is actual t value in sec
    varying_amplitude = rms * sin_wave
    for val in sin_wave.tolist():
        sin_values.append(val)
    return varying_amplitude

def concat_chunks(chunks):
    return np.concatenate(chunks)

def split_audio(audio_data, chunk_length, overlap):
    assert overlap < chunk_length, "Overlap must be less than chunk length"
    chunks = []
    start = 0
    length_of_audio = len(audio_data)
    while start < length_of_audio:
        end = start + chunk_length
        if end > len(audio_data):
            chunk = audio_data[start:length_of_audio]
        else:
            chunk = audio_data[start:end]
        chunks.append(chunk)
        start += chunk_length - overlap
    return chunks
    
    
def main():
    file_path = './elf_quote.wav'
    sample_rate, audio_data = wavfile.read(file_path)

    test_chunks = split_audio(audio_data, chunk_length=CHUNK_LENGTH, overlap=OVERLAP)
    filters = create_bandpass_filters(centre_frequency=CENTRE_FREQUENCY, band_width=BAND_WIDTH, filter_type=TYPE, number_of_filters=NUMBER_OF_FILTERS, interval=INTERVAL, order=ORDER)

    # list of lists where each list contains a channel corresponding to a specific filter and each channel contains the 
    # signal put through the given filter
    synth_chunks = []
    for chunk in test_chunks:
        chunks_for_filter = []
        for i, filter in enumerate(filters):
            start_time = i * CHUNK_LENGTH 
            filtered_chunk = filter_chunk(filter=filter, chunk=chunk)
            rms = get_rms(filtered_chunk=filtered_chunk)
            synth_chunk = synthesize(rms=rms, chunk_length=len(chunk), centre_freq_band=CENTRE_FREQUENCY + INTERVAL*i, start_time=start_time, sampling_rate=SAMPLING_RATE) 
            chunks_for_filter.append(synth_chunk)
        synth_chunks.append(np.sum(chunks_for_filter, axis=0))
    
    concatenated_chunks = np.concatenate(synth_chunks)
    
    sample_rate = 16e3
    num_channels = 1
    bytes_per_sample = 2 # 16 bit audio
    num_frames = len(concatenated_chunks)
    
    with wave.open(OUTPUT_FILE, 'w') as output_file:
        output_file.setparams((num_channels, bytes_per_sample, sample_rate, num_frames, "NONE", "Uncompressed"))
        output_file.writeframes(concatenated_chunks.astype(np.int16).tobytes())


    duration = len(audio_data) / sample_rate
    audio_time = np.linspace(0, duration, len(audio_data))
    plt.figure(figsize=(10, 5))
    plt.plot(audio_time, concatenated_chunks)
    plt.title("Synthesized Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(OUTPUT_GRAPH)
    
    plt.clf()
    
    plt.plot(rms_values)
    plt.savefig(OUTPUT_RMS_VALS)
    plt.clf()
    
    plt.plot(sin_values)
    plt.savefig(OUTPUT_SIN_VALS)
    plt.clf()
    
    df = pd.DataFrame({"synthesized_audio_data": concatenated_chunks})
    df.to_csv(f"{TITLE}_audio_data.csv")
    
    print("Finished")
            
        
if __name__ == "__main__":
    main()