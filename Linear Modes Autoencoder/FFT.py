import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

file_path = r"C:\Xingbo\University\McGill\Summer 2023\Research\Legrand\Code\Linear Modes Autoencoder\Modal Response data\2DOF.text"

file = open(file_path, "r")

Q_data_string = file.read()
count_list = Q_data_string.split(",")

Q_data = []
for i in range(len(count_list)):

    Q_data.append(float(count_list[i]))

# Sample time vs. displacement data
time = np.linspace(0, 100, 10000)  # Example time values (0 to 5 seconds)

# Calculate FFT
fft_result = fft(Q_data)
frequencies = fftfreq(len(time), time[1] - time[0])  # Calculate corresponding frequencies

# Plot the frequency vs. amplitude spectrum
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(fft_result))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Frequency vs. Amplitude Spectrum")
plt.grid()
plt.show()