import numpy as np
import soundfile as sf
import math
import tkinter as tk
from tkinter import filedialog
from tkinter.simpledialog import askinteger, askfloat
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def csch(x):
    """Compute the hyperbolic cosecant of x."""
    return 2 / (math.exp(x) - math.exp(-x))

def asinh(x):
    """Compute the inverse hyperbolic sine of x."""
    return math.log(x + math.sqrt(x * x + 1))

def normalh(x, length, a, b, c):
    """Compute the normalized hyperbolic weight."""
    n = b * c
    if x != 0:
        return asinh(x / (length / a * (1/b))) * csch(x / (length / a * c)) / n
    else:
        return 1  # The center coefficient

def generate_filter_coefficients(length, a, b, c, window_type=None):
    """Generate filter coefficients for the given length and parameters, optionally applying a window function."""
    if length % 2 == 0:
        raise ValueError("Length must be odd.")
    mid = length // 2
    weights = np.array([normalh(i - mid, length / 2, a, b, c) for i in range(length)])
    sum_weights = np.sum(weights)
    normalized_weights = weights / sum_weights

    # Apply window function if specified
    if window_type != "none":
        if window_type == 'hamming':
            window = np.hamming(length)
        elif window_type == 'blackman':
            window = np.blackman(length)
        elif window_type == 'kaiser':
            beta = askfloat("Input", "Enter the beta value for Kaiser window:")
            window = np.kaiser(length, beta)
        else:
            raise ValueError("Unsupported window type")
        # Applying the window to the filter coefficients
        normalized_weights *= window

    return normalized_weights

def normh_ma(source, length, a, b, c, window_type=None):
    """Apply a normalized hyperbolic moving average filter using generated coefficients."""
    if length % 2 == 0:
        raise ValueError("Length must be odd.")
    
    # Use the generate_filter_coefficients function to get the weights
    weights = generate_filter_coefficients(length, a, b, c, window_type)
    
    # Apply the convolution with the normalized weights
    filtered = np.convolve(source, weights, mode='same')
    return filtered

def process_audio_file(input_file, output_file, filter_length, a, b, c, window_type=None):
    """Read an audio file, apply the filter, and write the output to a new file."""
    data, samplerate = sf.read(input_file)
    if data.ndim > 1:
        filtered_channels = []
        for channel in range(data.shape[1]):
            filtered_channel = normh_ma(data[:, channel], filter_length, a, b, c, window_type)
            filtered_channels.append(filtered_channel)
        filtered_data = np.column_stack(filtered_channels)
    else:
        filtered_data = normh_ma(data, filter_length, a, b, c, window_type)
    
    sf.write(output_file, filtered_data, samplerate)

def browse_file():
    filename = filedialog.askopenfilename(title="Select an audio file",
                                          filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
    return filename

def display_filter_responses(weights, samplerate):
    """Display the frequency and phase response of the filter."""
    # Using a longer FFT length for better frequency resolution
    fft_length = max(4096, 2**int(np.ceil(np.log2(len(weights)))))
    w = np.fft.rfftfreq(fft_length, d=1/samplerate)
    h = np.fft.rfft(weights, n=fft_length)
    amplitude_response = 20 * np.log10(np.abs(h) + np.finfo(float).eps)  # Adding epsilon to avoid log(0)
    phase_response = np.angle(h, deg=True)

    plt.figure(figsize=(12, 6))
    
    # Subplot for Frequency Response
    plt.subplot(2, 1, 1)
    plt.semilogx(w, amplitude_response)
    plt.title('Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True, which='both', axis='both')
    plt.ylim(bottom=-96, top=0)  # Setting y-axis limits from -96 dB to 0 dB
    # Formatting frequency labels to display in decimal format
    formatter = FuncFormatter(lambda x, _: f'{x:.0f}')
    plt.gca().xaxis.set_major_formatter(formatter)

    # Subplot for Phase Response
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase_response)
    plt.title('Phase Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [Degrees]')
    plt.grid(True, which='both', axis='both')
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply the same formatter as the amplitude plot

    plt.tight_layout()
    plt.show()

def main():
    root = tk.Tk()
    root.title("Hyperbolic LPF")
    window_type = tk.StringVar(root)
    window_type.set("none")  # default value
    window_options = ["none", "hamming", "blackman", "kaiser"]
    window_dropdown = tk.OptionMenu(root, window_type, *window_options)
    window_dropdown.pack()

    def run_filter():
        print("Current window type:", window_type.get())  # Debugging line to check the value
        input_file = browse_file()
        if input_file:
            while True:
                _filter_length = askinteger("Input", "Enter FIR length:")
                if _filter_length is None:
                    break
                filter_length = _filter_length * 2 + 1
                a = askfloat("Input", "Enter a value:")
                if a is None:
                    break
                b = askfloat("Input", "Enter b value:")
                if b is None:
                    break
                c = askfloat("Input", "Enter c value:")
                if c is None:
                    break
                weights = generate_filter_coefficients(filter_length, a, b, c, window_type.get())
                samplerate = sf.info(input_file).samplerate
                display_filter_responses(weights, samplerate)

                if tk.messagebox.askyesno("Confirm Filter", "Is the filter setup okay?"):
                    output_file = filedialog.asksaveasfilename(title="Save the filtered audio as",
                                                               defaultextension=".wav",
                                                               filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
                    if output_file:
                        process_audio_file(input_file, output_file, filter_length, a, b, c, window_type.get())
                        tk.messagebox.showinfo("Success", "The audio has been processed successfully.")
                    break

    btn = tk.Button(root, text="Select Audio File and Apply Filter", command=run_filter)
    btn.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
