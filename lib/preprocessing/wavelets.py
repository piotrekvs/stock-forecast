import pywt, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def wavelet_denoise_series(series, wavelet='coif3', level=3, pad=100):
    data = series.values
    padded = np.pad(data, pad, mode='edge')
    coeffs = pywt.wavedec(padded, wavelet, mode='per', level=level)
    sigma  = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    thresh = sigma * np.sqrt(2 * np.log(len(padded)))
    coeffs[1:] = [pywt.threshold(c, thresh, 'soft') for c in coeffs[1:]]

    coeffs[-level] = np.zeros_like(coeffs[-level])

    denoised   = pywt.waverec(coeffs, wavelet, mode='per')[pad:-pad]
    return pd.Series(denoised[:len(series)], index=series.index)

def plot_wavelet_denoising(df, stock):
    # Print some metrics for verification
    snr = 10 * np.log10(np.sum(df['denoised'] ** 2) / np.sum(df['noise'] ** 2))
    print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    # Plotting the results
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
    fig.suptitle(f"Wavelet Denoising for {stock}")
    loc = 'upper left'

    # Plot Original Signal
    axes[0].plot(df.index, df['close'], label='Original Signal')
    axes[0].legend(loc=loc)
    axes[0].grid(False)  # Disable grid

    # Plot Denoised Signal
    axes[1].plot(df.index, df['denoised'], label='Denoised Signal')
    axes[1].legend(loc=loc)
    axes[1].grid(False)  # Disable grid

    # Plot Noise
    axes[2].plot(df.index, df['noise'], label='Extracted Noise')
    axes[2].legend(loc=loc)
    axes[2].grid(False)  # Disable grid


    plt.tight_layout() 
    plt.show()