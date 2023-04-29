import sys
import numpy as np
import scipy.io.wavfile as wav
import ntpath
from numpy.lib import stride_tricks
from matplotlib import pyplot as plt
import io
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import scipy.io.wavfile

output_folder = 'outputs'  # set your output folder and make sure it exists


# short-time Fourier Transformation(STFT)
def stft(sig, frame_size, overlap_factor=0.5, window=np.hanning):
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_factor * frame_size))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frame_size), strides=(
        samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def log_scale_spec(spec, sr=44100, factor=20.):
    time_bins, frequency_bins = np.shape(spec)
    scale = np.linspace(0, 1, frequency_bins) ** factor
    scale *= (frequency_bins-1)/max(scale)
    scale = np.unique(np.round(scale))
    # Creates spectrogram with new frequency bins
    new_spectrogram = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            new_spectrogram[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            new_spectrogram[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i+1])], axis=1)
    # Lists center frequency of bins
    all_frequencies = np.abs(np.fft.fftfreq(
        frequency_bins*2, 1./sr)[:frequency_bins+1])
    frequemcies = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            frequemcies += [np.mean(all_frequencies[int(scale[i]):])]
        else:
            frequemcies += [np.mean(all_frequencies[int(scale[i])                                                    :int(scale[i+1])])]
    return new_spectrogram, frequemcies


def plot_audio_spectrogram(audio_path, binsize=2**9, plot_path=None, argv='', colormap="jet"):
    # noise time * 44100 = will remove the noise form the signal

    sample_rate, samples = wav.read(audio_path)

    s = stft(samples, binsize)
    new_spectrogram, freq = log_scale_spec(s, factor=1.0, sr=sample_rate)
    data = 20. * np.log10(np.abs(new_spectrogram) / 1)  # dBFS
    x = data[:,410:425]
    x = np.sum(x, axis=1)
    #print(x.shape)
    #plt.plot(x)
    #plt.show()
    x = x[5:]
    x = x[:-100]

    mean = np.mean(x)
    #plt.plot(x)
    #plt.axhline(y=mean, color='r', linestyle='-')
    #plt.show()
    x = np.where(x >= mean, 1, 0)

    x = np.trim_zeros(x, 'f')
    x = np.trim_zeros(x, 'b')

    plt.plot(x)
    plt.show()
    t1 = x[:15400]
    t1 = np.trim_zeros(t1, 'f')
    t1 = np.trim_zeros(t1, 'b')
    t2 = x[15400:30800]
    t2 = np.trim_zeros(t2, 'f')
    t2 = np.trim_zeros(t2, 'b')
    t3 = x[30800:]
    t3 = np.trim_zeros(t3, 'f')
    t3 = np.trim_zeros(t3, 'b')

    t1 = np.array_split(t1, 152)
    t2 = np.array_split(t2, 152)
    t3 = np.array_split(t3, 152)
    tpzT1 = []
    tpzT2 = []
    tpzT3 = []
    for i, j, k in zip(t1, t2, t3):

        tpzT1.append(np.trapz(i))
        tpzT2.append(np.trapz(j))
        tpzT3.append(np.trapz(k))

    comb = [(x, y) for x in tpzT3 for y in tpzT2 ]
    possible_data = []
    for i in comb:
        x = np.where((np.array(tpzT3) <= i[0]) & (
            np.array(tpzT2) <= i[1])  , 0, 1)
        preamble = x[:12]
        postample = x[140:]
        if np.allclose(preamble, postample) == True and np.allclose(preamble, [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]):
            possible_data.append(x)


    import binascii
    import os
    from Crypto.Cipher import AES
    import time
    from binascii import unhexlify


    def decrypt_data(enc_data):
        enc_data = hex(int(enc_data, 2))
        enc_data = enc_data[2:]
        enc_data = unhexlify(enc_data)
        key = unhexlify('73656e696f725f70726f6a6563743230')

        iv = b'\xa3+\r\xb9\xd2x\xb9N\xf2\x8c\x94\xc4\x92d\x05\x1a'
        cipher = AES.new(key, AES.MODE_CBC, iv)
        plaintext = cipher.decrypt(enc_data)
        return plaintext


    def bin2text(s): return "".join(
        [chr(int(s[i:i+8], 2)) for i in range(0, len(s), 8)])


    def remove_ppabmle(bin_data):
        bin_data = bin_data[12:]
        bin_data = bin_data[:-12]
        return bin_data

    all_plaintext = []
    for i in possible_data:
        list = i.tolist()
        data = ''.join(str(e) for e in list)
        data = remove_ppabmle(data)
        plaintext = decrypt_data(data)

        all_plaintext.append(plaintext)
    np.savetxt("data.txt", all_plaintext, fmt='%s',newline=" ",)
    '''
    #print(data.shape)
    #np.savetxt("data.csv", data/data.max(), delimiter=",")
    time_bins, freq_bins = np.shape(data)
    print("Time bins: ", time_bins)
    print("Frequency bins: ", freq_bins)
    print("Sample rate: ", sample_rate)
    print("Samples: ", len(samples))
    # horizontal resolution correlated with audio length  (samples / sample length = audio length in seconds). If you use this(I've no idea why). I highly recommend to use "gaussian" interpolation.
    #plt.figure(figsize=(len(samples) / sample_rate, freq_bins / 100))
    # resolution equal to audio data resolution, dpi=100 as default
    plt.figure(figsize=(time_bins/200, freq_bins/200))

    plt.imshow(np.transpose(data), origin="lower", aspect="auto",
               cmap=colormap, interpolation="none")
    # Labels
    plt.xlabel("Time(s)")
    plt.ylabel("Frequency(Hz)")
    plt.xlim([0, time_bins-1])
    plt.ylim([0, freq_bins])
    if 'l' in argv:  # Add Labels
        plt.colorbar().ax.set_xlabel('dBFS')
    else:  # No Labels
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
    
    x_locations = np.float32(np.linspace(0, time_bins-1, 30))
    plt.xticks(x_locations, ["%.02f" % l for l in (
        (x_locations*len(samples)/time_bins)+(0.5*binsize))/sample_rate])
    y_locations = np.int16(np.round(np.linspace(0, freq_bins-1, 20)))
    plt.yticks(y_locations, ["%.02f" % freq[i] for i in y_locations])
    
    if 's' in argv:  # Save
        print('Unlabeled output saved as.png')
        plt.savefig(plot_path)
    else:
        print('Graphic interface...')
        mplcursors.cursor() # or just mplcursors.cursor()

        plt.show()
    '''
    return data


if len(sys.argv) > 2:
    ims = plot_audio_spectrogram(sys.argv[1], 2**10, output_folder + '/' + ntpath.basename(
        sys.argv[1].replace('.wav', '')) + '.png',  sys.argv[2])
else:
    ims = plot_audio_spectrogram(sys.argv[1], 2**10, None, '')
