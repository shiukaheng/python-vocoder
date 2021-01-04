import pyaudio
from scipy.io.wavfile import read, write
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from datetime import timezone, datetime

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
    # peak_indices = sig.find_peaks(mag, distance=20)[0]
    # cs = interp.CubicSpline(peak_indices, [mag[x] for x in peak_indices])
    # envelope = cs(np.linspace(0, len(mag)-1, len(mag)))
    envelope = sig.savgol_filter(mag, 101, 1)
    envelope = envelope - np.min(envelope)
    envelope = envelope / np.max(envelope)
    envelope = envelope / np.sum(envelope) * summag
    # plt.plot(envelope)
    # plt.gca().set_xlim([0, 1000])
    # plt.gca().set_ylim([0, 1])
    # plt.show()
    # plt.clf()
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


def extract_filter(file, plot=False):
    r = proc_mag_only(stft(file), envelope_proc)
    if plot:
        nmag = 20 * np.log10(r['mag'])
        for mag in nmag:
            plt.plot(r['frequency'], mag)
            plt.gca().set_xlim([0, 8000])
            plt.gca().set_ylim([-100, 0])
            plt.title("Filter plot")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Level [dB]")
            plt.savefig("filterplot/"+str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))+".png")
            plt.clf()
    return r


def extract_source(file, plot=False):
    r = process(file, deenvelope_proc)
    if plot:
        nft = 20 * np.log10(r['ft'])
        for ft in np.transpose(nft):
            plt.plot(r['frequency'], ft)
            plt.gca().set_xlim([0, 8000])
            plt.gca().set_ylim([-100, 0])
            plt.title("Source plot")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Level [dB]")
            plt.savefig("sourceplot/" + str(int(datetime.now(tz=timezone.utc).timestamp() * 1000)) + ".png")
            plt.clf()
    return r


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

def mvo(stft, path="/", title=""):
    logft = 20 * np.log10(stft['ft'])
    for ft in np.transpose(logft):
        plt.plot(stft['frequency'], ft)
        plt.gca().set_xlim([0, 8000])
        plt.gca().set_ylim([-100, 0])
        if title != "":
            plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Level [dB]")
        plt.savefig(path + str(int(datetime.now(tz=timezone.utc).timestamp() * 1000)) + ".png")
        plt.clf()


f = read_file("google_2.wav")

# src, comp = vocoder(f)
# save(src)
# play(comp)
# # play(comp)
# g = read_file("sample.wav")
s = read_file("carrier.wav")
# fa = stft(g)
# mvo(fa, "resynthesized/", "Resynthesized plot")
comp2 = musical_vocoder(s, f)
save(comp2)
# play(comp2)
