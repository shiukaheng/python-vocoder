import pyaudio
from scipy.io.wavfile import read, write
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import pandas as pd


def example_reader(path):
    a = read(path)
    data = np.array(a[1], dtype=float)
    data = data/np.max([np.max(data), np.min(data)])
    return {'waveform': data, 'rate': a[0]}


def stft(read_file, chunk=2048, nfft=8192):
    f, t, zxx = sig.stft(read_file['waveform'], read_file['rate'], return_onesided=True, nperseg=chunk, nfft=nfft)
    return {'frequency': f, 'chunk_time': t, 'ft': zxx, 'rate': read_file['rate'], 'chunk': chunk, 'nfft': nfft}


def proc_stft_amp(stft, function):
    mag_list = []
    ft_list = []
    for chunk in np.transpose(stft['ft']):
        mag = np.abs(chunk)
        phase = np.divide(chunk, mag)
        ft_list.append(np.multiply(phase, function(mag, stft['rate'])))
    nft = np.transpose(np.stack(ft_list))
    return {'frequency': stft['frequency'], 'chunk_time': stft['chunk_time'], 'ft': nft, 'rate': stft['rate'], 'chunk': stft['chunk'], 'nfft': stft['nfft']}


def istft(stft):
    waveform = sig.istft(stft['ft'], fs=stft['rate'], nperseg=stft['chunk'], nfft=stft['nfft'], input_onesided=True)
    return {'waveform': waveform[1], 'rate': stft['rate']}


def play(read_file, play_samplerate=44100, out_index=4):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=play_samplerate,
                    frames_per_buffer=1024,
                    output=True,
                    output_device_index=out_index
                    )
    if read_file['rate'] != play_samplerate:
        wf = sig.resample(read_file['waveform'], play_samplerate*read_file['waveform'].size/read_file['rate'])
    else:
        wf = read_file['waveform']
    wf = wf/np.max(np.abs(wf))*32767
    stream.write(wf.astype(np.int16).tostring())
    stream.close()


def process(data, function, chunk=2048):
    return istft(proc_stft_amp(stft(data, chunk=chunk), function))


def dummy_proc(mag, rate):
    return mag


def sharpen_proc(mag, rate):
    r = np.power(mag, 2)
    r = r/np.max(r)*np.max(mag)
    return r


def envelope_proc(mag, rate):
    # summag = np.sum(mag)
    # peak_indices = sig.find_peaks(mag, distance=20)[0]
    # cs = interp.CubicSpline(peak_indices, [mag[x] for x in peak_indices])
    # envelope = cs(np.linspace(0, len(mag)-1, len(mag)))
    envelope = sig.savgol_filter(mag, 101, 1)
    return envelope


def deenvelope_proc(mag, rate):
    o_summag = np.sum(mag)
    envelope = envelope_proc(mag, rate)
    nmag = mag/envelope
    nmag = nmag/np.sum(nmag)*o_summag
    return nmag


def visualize(data, function, i=200, chunk=1024):
    o_ft = stft(data, chunk=chunk)
    ft = proc_stft_amp(o_ft, function)
    chunk = np.transpose(ft['ft'])[i]
    o_chunk = np.transpose(o_ft['ft'])[i]

    mag = np.abs(chunk)
    o_mag = np.abs(o_chunk)

    plt.plot(ft['frequency'], mag)
    plt.plot(ft['frequency'], o_mag)

    plt.gca().set_xlim([0, 8000])
    plt.yscale('log')
    plt.show()


def waveform(data):
    plt.plot(data['waveform'])
    plt.show()


def save(data, name='sample.wav'):
    wf = data['waveform']
    wf = wf / np.max(np.abs(wf))
    write(name, data['rate'], wf)


data = example_reader("test2.wav")
visualize(data, deenvelope_proc, chunk=2048, i=30)
visualize(data, envelope_proc, chunk=2048, i=30)
ndata = process(data, deenvelope_proc)
play(ndata)
save(ndata)