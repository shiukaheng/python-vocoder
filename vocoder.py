# import pyaudio
from scipy.io.wavfile import read, write
import scipy.signal as sig
import numpy as np
import scipy.interpolate as interp
import sys
import os.path
from os import path

def resample(file, target_rate):
    if file['rate'] != target_rate:
        return {'waveform':sig.resample(file['waveform'], target_rate*file['waveform'].size/file['rate']), 'rate': target_rate}
    else:
        return file


def read_file(path):
    a = read(path)
    data = np.array(a[1], dtype=float)
    data = data/np.max([np.max(data), np.min(data)])
    return {'waveform': data, 'rate': a[0]}


def stft(read_file, chunk=2048, nfft=8192):
    f, t, zxx = sig.stft(read_file['waveform'], read_file['rate'], return_onesided=True, nperseg=chunk, nfft=nfft)
    return {'frequency': f, 'chunk_time': t, 'ft': zxx, 'rate': read_file['rate'], 'chunk': chunk, 'nfft': nfft}


def istft(stft):
    waveform = sig.istft(stft['ft'], fs=stft['rate'], nperseg=stft['chunk'], nfft=stft['nfft'], input_onesided=True)
    return {'waveform': waveform[1], 'rate': stft['rate']}


def proc_stft_amp(stft, function):
    ft_list = []
    for chunk in np.transpose(stft['ft']):
        mag = np.abs(chunk)
        phase = np.divide(chunk, mag)
        ft_list.append(np.multiply(phase, function(mag, stft['rate'])))
    nft = np.transpose(np.stack(ft_list))
    return {'frequency': stft['frequency'], 'chunk_time': stft['chunk_time'], 'ft': nft, 'rate': stft['rate'], 'chunk': stft['chunk'], 'nfft': stft['nfft']}


def musical_vocoder(source, filter, output_image=False):
    filter_data = extract_filter(filter)
    source_data = stft(resample(source, filter_data['rate']), chunk=filter_data['chunk'])
    # mvo(source_data, path="musicalsourceplot/", title="Source plot")
    source_ft = np.transpose(source_data['ft'])[:np.shape(filter_data['mag'])[0]]
    filter_mag = filter_data['mag'][:np.shape(source_ft)[0]]
    r = np.stack([np.multiply(z, f) for z, f in zip(np.transpose(source_ft), np.transpose(filter_mag))])
    new_stft = {'frequency': filter_data['frequency'], 'ft': r, 'rate': filter_data['rate'], 'chunk': filter_data['chunk'], 'nfft': filter_data['nfft']}
    return istft(new_stft)

def process(data, function, chunk=2048):
    return proc_stft_amp(stft(data, chunk=chunk), function)

def proc_mag_only(stft, function):
    mag_list = []
    for chunk in np.transpose(stft['ft']):
        mag = np.abs(chunk)
        mag_list.append(function(mag, stft['rate']))
    mag_list = np.stack(mag_list)
    mag_list = mag_list/np.max(mag_list)
    return {'frequency': stft['frequency'], 'chunk_time': stft['chunk_time'], 'mag': mag_list, 'rate': stft['rate'], 'chunk': stft['chunk'], 'nfft': stft['nfft']}


def envelope_proc(mag, rate):
    summag = np.sum(mag)
    envelope = sig.savgol_filter(mag, 101, 1)
    envelope = envelope - np.min(envelope)
    envelope = envelope / np.max(envelope)
    envelope = envelope / np.sum(envelope) * summag
    return envelope

def raw_envelope_proc(mag, rate):
    envelope = sig.savgol_filter(mag, 101, 1)
    return envelope

def deenvelope_proc(mag, rate):
    o_summag = np.sum(mag)
    envelope = raw_envelope_proc(mag, rate)
    nmag = np.divide(mag, envelope)
    nmag = nmag/np.sum(nmag)*o_summag
    return nmag

def extract_filter(file):
    r = proc_mag_only(stft(file), envelope_proc)
    return r

def extract_source(file):
    r = process(file, deenvelope_proc)
    return r

def save(data, name='sample.wav'):
    wf = data['waveform']
    wf = wf / np.max(np.abs(wf))
    write(name, data['rate'], wf)

def vocoder(file):
    filter_data = extract_filter(file)
    source_data = extract_source(file)
    source_ft = np.transpose(source_data['ft'])[:np.shape(filter_data['mag'])[0]]
    filter_mag = filter_data['mag'][:np.shape(source_ft)[0]]
    r = np.stack([np.multiply(z, f) for z, f in zip(np.transpose(source_ft), np.transpose(filter_mag))])
    new_stft = {'frequency': filter_data['frequency'], 'ft': r, 'rate': filter_data['rate'],
                'chunk': filter_data['chunk'], 'nfft': filter_data['nfft']}
    return istft(source_data), istft(new_stft)

if __name__ == "__main__":
    if not (path.exists(sys.argv[1]) and path.exists(sys.argv[2]) and (not path.exists(sys.argv[3]))):
        print("Usage:")
        print("python vocoder.py <carrier file> <modulator file> <output file>")
        exit()
    print("Reading carrier...")
    carrier = read_file(sys.argv[1])
    print("Reading modulator...")
    modulator = read_file(sys.argv[2])
    print("Processing...")
    result = musical_vocoder(carrier, modulator)
    save(result, name=sys.argv[3])
    print("Done!")