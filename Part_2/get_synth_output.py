import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import wave

# TEST BASE PARAMETERS
CHUNK_LENGTH = 160 # 160 is 10ms because sample rate is 16kHz
OVERLAP = 0
CENTRE_FREQUENCY = 1000
SAMPLING_RATE = 16e3
TYPE = "chebyshev"
BAND_WIDTH = CENTRE_FREQUENCY-10
ORDER = 40
NUMBER_OF_FILTERS = 3
INTERVAL = 40
OUTPUT_FILE = "./output_test.wav"

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
        if filter_type == "butterworth":
            return signal.butter(
                order, 
                [centre_freq - band_width, centre_freq + band_width], 
                btype='bandpass', 
                output='sos', 
                fs=sample_rate
            )
        elif filter_type == "chebyshev":
            return signal.cheby2(
                order, 
                20, 
                [centre_freq - band_width, centre_freq + band_width], 
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

def synthesize(rms, chunk_length, centre_freq_band, sampling_rate=SAMPLING_RATE):
    time = np.arange(chunk_length)
    sin_wave = np.sin(2 * np.pi * centre_freq_band * time/sampling_rate)
    varying_amplitude = rms * sin_wave
    for val in varying_amplitude.tolist():
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
    for filter in filters:
        chunks_for_filter = []
        for i, chunk in enumerate(test_chunks):
            filtered_chunk = filter_chunk(filter=filter, chunk=chunk)
            rms = get_rms(filtered_chunk=filtered_chunk)
            # idk if im correctly understanding this equation
            synth_chunk = synthesize(rms=rms, chunk_length=len(chunk), centre_freq_band=CENTRE_FREQUENCY + INTERVAL*i) 
            chunks_for_filter.append(synth_chunk)
            # chunks_for_filter.append(filtered_chunk)
        synth_chunks.append(chunks_for_filter)

    concatenated_chunks = []
    for chunk_channel in synth_chunks:
        concatenated_chunks.append(concat_chunks(chunk_channel))
    
    concatenated_chunks = np.array(concatenated_chunks)

    summed_chunks = np.sum(concatenated_chunks, axis=0)    
    
    sample_rate = 16e3
    num_channels = 1
    bytes_per_sample = 2 # 16 bit audio
    num_frames = len(summed_chunks)
    
    with wave.open(OUTPUT_FILE, 'w') as output_file:
        output_file.setparams((num_channels, bytes_per_sample, sample_rate, num_frames, "NONE", "Uncompressed"))
        output_file.writeframes(summed_chunks.astype(np.int16).tobytes())
    
    amplified_chunks = summed_chunks*10e11
        
    with wave.open("output_amplified.wav", "w") as outputfile:
        outputfile.setparams((num_channels, bytes_per_sample, sample_rate, num_frames, "NONE", "Uncompressed"))
        outputfile.writeframes(amplified_chunks.astype(np.int16).tobytes())
        
    plt.plot(summed_chunks)
    plt.savefig("./output_test.png")
    plt.clf()
    
    plt.plot(amplified_chunks)
    plt.savefig("./output_amplified.png")   
    plt.clf()
    
    plt.plot(rms_values)
    plt.savefig("./rms_values.png")
    plt.clf()
    
    plt.plot(sin_values)
    plt.savefig("./sin_values.png")
    plt.clf()
    
    print("Finished")
            
        
if __name__ == "__main__":
    main()