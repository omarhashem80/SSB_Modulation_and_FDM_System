import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
import os

class SSB_FDM:
    """
    A class for Single Sideband (SSB) modulation and demodulation,
    and Frequency Division Multiplexing (FDM).
    
    This class provides methods to record audio, apply low-pass filtering,
    perform SSB modulation/demodulation, and visualize signal spectra.
    """

    def __init__(self, sample_rate=44100, record_duration=10, carrier_frequencies=None, lpf_cutoff=2250):
        """
        Initialize the SSB_FDM class with parameters.

        Args:
            sample_rate (int): Sampling rate in Hz.
            record_duration (int): Recording duration in seconds.
            carrier_frequencies (list): List of carrier frequencies for modulation.
            lpf_cutoff (int): Low-pass filter cutoff frequency in Hz.
        """
        # Initialize the parameters with default values or user inputs
        self.SAMPLE_RATE = sample_rate
        self.RECORD_DURATION = record_duration
        self.CARRIER_FREQUENCIES = carrier_frequencies or [5000, 10000, 15000]
        self.LPF_CUTOFF = lpf_cutoff

    def record_audio(self, filename):
        """
        Record audio and save it as a WAV file. If the file exists, load the existing file.

        Args:
            filename (str): Path to save or load the WAV file.

        Returns:
            np.ndarray: The audio signal data.
        """
        # Check if the file already exists
        if os.path.exists(filename):
            print(f"{filename} already exists. Loading the existing file.")
            samplerate, data = wavfile.read(filename)
            if samplerate != self.SAMPLE_RATE:
                raise ValueError("Sample rate mismatch!")
            if len(data.shape) > 1:  # Convert stereo to mono if necessary
                data = data[:, 0]
            return data
        else:
            # Record audio if the file does not exist
            print(f"Recording {filename} for {self.RECORD_DURATION} seconds...")
            recording = sd.rec(int(self.RECORD_DURATION * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()  # Wait for the recording to finish
            recording = np.squeeze(recording)
            wavfile.write(filename, self.SAMPLE_RATE, recording)
            print(f"Saved {filename}")
            return recording

    def apply_low_pass_filter(self, signal_data):
        """
        Apply a Butterworth low-pass filter to the signal.

        Args:
            signal_data (np.ndarray): Input signal.

        Returns:
            np.ndarray: Low-pass filtered signal.
        """
        nyquist = self.SAMPLE_RATE / 2
        normalized_cutoff = self.LPF_CUTOFF / nyquist
        filter_length = 101
        n = np.arange(filter_length) - (filter_length - 1) / 2
        sinc_filter = np.sinc(2 * normalized_cutoff * n)
        window = np.hamming(filter_length)
        filter_coeffs = sinc_filter * window
        filter_coeffs /= np.sum(filter_coeffs)
        filtered_signal = np.convolve(signal_data, filter_coeffs, mode='same')
        return filtered_signal

    def normalize_signal(self, signal_data):
        """
        Normalize the signal to the range [-1, 1].

        Args:
            signal_data (np.ndarray): Input signal.

        Returns:
            np.ndarray: Normalized signal.
        """
        return signal_data / np.max(np.abs(signal_data))

    def half_transform(self, x):
        """
        Perform the Hilbert Transform to generate an analytic signal.

        Args:
            x (np.ndarray): Input signal.

        Returns:
            np.ndarray: Imaginary part of the analytic signal.
        """
        N = len(x)
        X_f = np.fft.fft(x)
        H = np.zeros(N)
        H[0] = 1
        H[1:(N // 2)] = 2  # Double amplitude to fix modulation.
        if N % 2 == 0:
            H[N // 2] = 1  # Nyquist frequency for even-length signals
        analytic_signal = np.fft.ifft(X_f * H)
        return np.imag(analytic_signal)

    def ssb_modulation(self, signal_data, carrier_freq):
        """
        Perform Single Sideband (SSB) modulation.

        Args:
            signal_data (np.ndarray): Input signal.
            carrier_freq (float): Carrier frequency in Hz.

        Returns:
            np.ndarray: SSB modulated signal.
        """
        t = np.linspace(0, len(signal_data) / self.SAMPLE_RATE, len(signal_data), endpoint=False)
        analytic_signal = self.half_transform(signal_data)
        carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
        carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
        modulated_signal = signal_data * carrier_cos - analytic_signal * carrier_sin
        return modulated_signal

    def ssb_demodulation(self, modulated_signal, carrier_freq):
        """
        Perform Single Sideband (SSB) demodulation.

        Args:
            modulated_signal (np.ndarray): Modulated input signal.
            carrier_freq (float): Carrier frequency in Hz.

        Returns:
            np.ndarray: Demodulated signal.
        """
        t = np.linspace(0, len(modulated_signal) / self.SAMPLE_RATE, len(modulated_signal), endpoint=False)
        carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
        carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
        demod_cos = modulated_signal * carrier_cos
        demod_sin = modulated_signal * carrier_sin
        demod_cos_filtered = self.apply_low_pass_filter(demod_cos)
        demod_sin_filtered = self.apply_low_pass_filter(demod_sin)
        reconstructed_signal = demod_cos_filtered + 1j * demod_sin_filtered
        return self.normalize_signal(np.real(reconstructed_signal))

    def plot_magnitude_spectrum(self, signal_data, title):
        """
        Plot the magnitude spectrum of the signal.

        Args:
            signal_data (np.ndarray): Input signal.
            title (str): Title of the plot.
        """
        freq_spectrum = np.fft.fft(signal_data)
        freq_axis = np.fft.fftfreq(len(signal_data), 1 / self.SAMPLE_RATE)
        plt.plot(freq_axis[:len(freq_axis)//2], np.abs(freq_spectrum[:len(freq_spectrum)//2]))
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

    def process(self, input_files, output_files):
        """
        Record, process, modulate, and demodulate signals.

        Args:
            input_files (list): List of input WAV filenames.
            output_files (list): List of output WAV filenames for demodulated signals.
        """
        # Step 1: Record or Load Audio
        input_signals = []
        for filename in input_files:
            signal_data = self.record_audio(filename)
            input_signals.append(signal_data)

        # Step 2: Filter, Modulate, and Plot Spectra
        filtered_signals = []
        modulated_signals = []

        # Increase the figure size to avoid overlapping axes
        plt.figure(figsize=(18, 14))  # Larger figure size
        for i, (signal_data, carrier_freq) in enumerate(zip(input_signals, self.CARRIER_FREQUENCIES), start=1):
            # Plot original signal spectrum
            plt.subplot(4, 3, 3 * i - 2)
            self.plot_magnitude_spectrum(signal_data, f"Original Signal {i} Spectrum")

            # Apply low-pass filter
            filtered_signal = self.apply_low_pass_filter(signal_data)
            filtered_signals.append(filtered_signal)

            # Plot filtered signal spectrum
            plt.subplot(4, 3, 3 * i - 1)
            self.plot_magnitude_spectrum(filtered_signal, f"Filtered Signal {i} Spectrum")

            # SSB modulation
            modulated_signal = self.ssb_modulation(filtered_signal, carrier_freq)
            modulated_signals.append(modulated_signal)

            # Plot modulated signal spectrum
            plt.subplot(4, 3, 3 * i)
            self.plot_magnitude_spectrum(modulated_signal, f"Modulated Signal {i} Spectrum")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Step 3: Combine modulated signals for FDM
        min_length = min(len(signal) for signal in modulated_signals)
        modulated_signals_trimmed = [signal[:min_length] for signal in modulated_signals]
        fdm_signal = np.sum(modulated_signals_trimmed, axis=0)

        # Plot the FDM signal spectrum
        plt.figure(figsize=(12, 8))  # Larger figure size for the FDM signal spectrum
        self.plot_magnitude_spectrum(fdm_signal, "FDM Signal Spectrum")
        plt.title("Frequency Division Multiplexing (FDM) Spectrum")
        plt.tight_layout()

        # Step 4: Demodulate and save the results
        for i, (modulated_signal, carrier_freq, output_file) in enumerate(zip(modulated_signals, self.CARRIER_FREQUENCIES, output_files), start=1):
            # Perform SSB demodulation
            demodulated_signal = self.ssb_demodulation(modulated_signal, carrier_freq)
            wavfile.write(output_file, self.SAMPLE_RATE, demodulated_signal.astype(np.float32))
            print(f"Demodulated signal {i} saved to {output_file}")
            
            self.plot_magnitude_spectrum(demodulated_signal, f"Demodulated Signal {i} Spectrum")
            plt.show()
if __name__ == "__main__":
    # Define the base directory for input/output files
    base = './content/'
    
    # List of input files to be processed
    input_files = [base + 'input1.wav', base + 'input2.wav', base + 'input3.wav']
    output_files = ['output1.wav', 'output2.wav', 'output3.wav']

    # Initialize the SSB_FDM processor and run the process method
    processor = SSB_FDM()
    processor.process(input_files, output_files)