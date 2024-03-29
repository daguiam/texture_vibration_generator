import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import wave
import pyaudio
import time



import win32gui
import win32api

import math
def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


class Cursor():
    def __init__(self,dpi=1600):
        self.prev_time = time.time()
        self.prev_pos = self.position()
        self.prev_speed = 0
        self.dpi = dpi
        self.dpmm = self.dpi*np.sqrt(3.93701)
        self.dpmm = self.dpi*25.4
            
    def position(self):
        (x,y) = win32gui.GetCursorPos()
        return np.array([x,y])
        
    def speed(self):
        """ Returns speed in pixels/second"""
        
        curr_pos = self.position()
        curr_time = time.time()
        delta_pos = curr_pos-self.prev_pos
        delta_time = curr_time-self.prev_time

        if delta_time == 0:
            speed = self.prev_speed
        else:
            speed = delta_pos/delta_time
#         print(delta_pos, speed)
        speed = np.linalg.norm(speed)
        self.prev_pos = curr_pos
        self.prev_time = curr_time
        self.prev_speed = speed
        return speed
        


from collections import deque

class CircularBuffer (deque):
    """ Creates a circular buffer class which inherits the deque collection 
    and implements extract functions"""
    
    def extract(self,n):
        """ Extracts n items from the right """
        # return list([self.pop() for i in range(n)])
        return list(reversed([self.pop() for i in range(n)]))
    
    def extractleft(self, n):
        """ Extracts n items from the left """
        return list([self.popleft() for i in range(n)])
        # return list(reversed([self.popleft() for i in range(n)]))
    
        
        


def resample_spectrum(spectrum, ratio):
    N = len(spectrum)
    new_N = np.int(N*ratio)
    spectrum = np.fft.fftshift(spectrum)
    
    if ratio>1:
        # pad beginning and end with zeros
        extra_zeros = np.int((new_N-N)/2)
        
        spectrum = np.append(np.zeros(extra_zeros), spectrum,)
        spectrum = np.append(spectrum,np.zeros(extra_zeros))
        
    spectrum = np.fft.fftshift(spectrum)
    return spectrum  




def generate_audio_from_spectrum(spectrum,  N_output=None):
    """
    From a reference spatial spectrum, generate the audio signal of a dirac probe sweeping at a velocity
    over this spectrum, for a given sample rate (fs_output) and for N_output data points
    spectrum : reference spectrum
    N_output : number of samples for the output signal
    """
    
    if N_output is not None:
        N = len(spectrum)
        spectrum = spectrum_texture
        
        fs_ratio = N_output/N
        
        
        
        # padding the spectrum data to resample
        spectrum = np.fft.fftshift(spectrum)
        padding = N_output-N
        spectrum = np.pad(spectrum, np.int(padding/2), 'constant')
    spectrum = np.fft.fftshift(spectrum)
    sig = np.fft.ifft(spectrum, norm='ortho')
    return sig

def change_pitch(sig, ratio, start_phase=0):
    """ Interpolates the signal sig between 0 and 1 
    and multiplies this sampling to a ratio of its sampling,
    starting with a start_phase between 0-1.
    """
    sig_x = np.linspace(0,1,len(sig))

    sig_dx = np.diff(sig_x)[0]
    sig_interp = interpolate.interp1d(sig_x, sig, bounds_error=True)
    new_x = np.arange(len(sig_x))*sig_dx*ratio+start_phase
    new_sig = sig_interp((new_x)%1)
    
    return new_sig



def array2audio(note,max_amplitude=2):
    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / max_amplitude
    # Convert to 16-bit data
    audio = audio.astype(np.int16)
    return audio

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass_coefficients(cutoff, fs, order=5):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b,a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def estimate_texture_signal(spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, ):
    """ Estimates the audio signal, sampled at fs_radio, obtained from a reference spectrum texture
    spectrum_texture : array, reference spatial spectrum of the spectrum
    fs_spatial : spatial sampling frequency of the texture
    velocity_probe : velocity of the probe over the spectrum texture
    N_audio_segment : number of samples of the output audio segment
    fs_audio : sampling frequency of the output audio signal
    
    """
    sample_time_span = N_audio_segment/fs_audio  # time of each audio_segment
    N_frame = N_audio_segment
    if velocity_probe ==0 :
        #print("zero velocity")
        sig_frame = np.zeros(N_frame)
        t_frame = np.linspace(0,sig_frame.size/fs_audio, sig_frame.size, endpoint=False)

    else:

        # first get signal from texture sampled at fs_spatial
        N_texture = len(spectrum_texture)
        sig_spatial = generate_audio_from_spectrum(spectrum_texture, N_output=None)

        N_spatial = len(sig_spatial)
        t_spatial = np.linspace(0, N_texture/fs_spatial, N_spatial, endpoint=False)


        # resample the signal to the fs_temporal = fs_spatial*velocity_probe

        sig_temporal = sig_spatial
        N_temporal = len(sig_temporal)

        fs_temporal = fs_spatial*velocity_probe
        window_time_span = N_spatial/fs_temporal

        # repeats the transient signal to fit the sample_time_span 
        # if window_time_span < sample_time_span:
        #     reps = np.int(np.ceil(sample_time_span/window_time_span))
        #     #print("reps",reps)

        #     sig_temporal = np.tile(sig_temporal, reps)
        #     N_temporal = len(sig_temporal)


        # Resample the temporal signal to the fs_audio sampling rate

        N_audio = np.int(fs_audio/fs_temporal*N_temporal)
        t_temporal = np.linspace(0, N_temporal/fs_temporal,N_temporal, endpoint=False)
        sig_audio = sig_temporal
        t_audio = t_temporal

        sig_audio, t_audio = signal.resample(sig_temporal, N_audio, t=t_temporal, window='hamming')
        
        
        
        # Do not trim
        # Trims the audio signal to the sample_time_span

        
        # sig_frame = sig_audio[0:N_frame]
        sig_frame = sig_audio
        t_frame = t_audio
    # N_frame = sig_frame.size
    
    return (t_frame, sig_frame.real)



def estimate_texture_signal_constantframesize(spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, ):
    """ Estimates the audio signal, sampled at fs_radio, obtained from a reference spectrum texture
    spectrum_texture : array, reference spatial spectrum of the spectrum
    fs_spatial : spatial sampling frequency of the texture
    velocity_probe : velocity of the probe over the spectrum texture
    N_audio_segment : number of samples of the output audio segment
    fs_audio : sampling frequency of the output audio signal
    
    """
    sample_time_span = N_audio_segment/fs_audio  # time of each audio_segment
    N_frame = N_audio_segment
    if velocity_probe ==0 :
        #print("zero velocity")
        sig_frame = np.zeros(N_frame)
    else:

        # first get signal from texture sampled at fs_spatial
        N_texture = len(spectrum_texture)
        sig_spatial = generate_audio_from_spectrum(spectrum_texture, N_output=None)

        N_spatial = len(sig_spatial)
        t_spatial = np.linspace(0, N_texture/fs_spatial, N_spatial)


        # resample the signal to the fs_temporal = fs_spatial*velocity_probe

        sig_temporal = sig_spatial
        N_temporal = len(sig_temporal)

        fs_temporal = fs_spatial*velocity_probe
        window_time_span = N_spatial/fs_temporal

        # repeats the transient signal to fit the sample_time_span 
        if window_time_span < sample_time_span:
            reps = np.int(np.ceil(sample_time_span/window_time_span))
            #print("reps",reps)

            sig_temporal = np.tile(sig_temporal, reps)
            N_temporal = len(sig_temporal)


        # Resample the temporal signal to the fs_audio sampling rate

        N_audio = np.int(fs_audio/fs_temporal*N_temporal)
        sig_audio = signal.resample(sig_temporal, N_audio)
        t_audio = np.linspace(0,N_audio/fs_audio, N_audio)


        # Trims the audio signal to the sample_time_span

        sig_frame = sig_audio[0:N_frame]
        # sig_frame = sig_audio
    # N_frame = sig_frame.size
    t_frame = np.linspace(0,N_frame/fs_audio, N_frame)
    
    return (t_frame, sig_frame.real)


def window_easing(n, easing=None):
    """
    n - int, window size
    easing - int, samples to attenuate at edges with sine"""
    if easing is None:
        easing = np.int(np.floor(n/2))
    
    assert easing <= n/2, "easing must be less than half the window size"
    window = np.ones(n)
    if easing>0:
        #window[:easing] = np.sin(2*np.pi*(np.arange(easing)/(2*easing)-1/4))/2+0.5
        #window[-easing:] = np.sin(2*np.pi*(np.arange(easing)/(2*easing)+1/4))/2+0.5
        window[:easing] = np.linspace(0,1,easing, endpoint=True)
        window[-easing:] = np.flip(window[:easing])
    return window



def append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap ):

    N_frame = N_audio_segment 
    N_frame = N_audio_segment + N_overlap
    t_frame, sig_frame = estimate_texture_signal(spectrum_texture, fs_spatial, velocity_probe, N_frame, fs_audio, )
    
    N_frame = sig_frame.size

    # print("Nframe",N_frame)
    
    window = window_easing(N_frame, N_overlap)
    # Remove the overlap samples
    sig_frame = sig_frame*window


    # print("sig frame 1",len(sig_frame), len(buffer))
    if len(buffer)>N_overlap:
        tail = np.array(buffer.extract(N_overlap))

        sig_frame[:N_overlap] = sig_frame[:N_overlap]+tail
        # print("sig frame 2",len(sig_frame), len(tail))
#     print(sig_frame)
    buffer.extend(list(sig_frame.real))
    return sig_frame.size






# define callback (2)
def callback(in_data, frame_count, time_info, status):

    global zf
    global b,a
    global buffer
    global next_frame
        
    
    frame = np.copy(buffer.extractleft(frame_count))
    
#         sig = butter_lowpass_filter(sig, cutoff, fs_audio, order=5)
    # frame, zf = signal.lfilter(b, a, frame, zi=zf)
    
    
    audio = array2audio(frame.real)
    data = audio
    
    output_time = time_info['output_buffer_dac_time']
    current_time = time_info['current_time']
#     print(N_ratio, sampling_ratio)
#     print("\r %0.3f    fs=%0.3fHz    %0.3fmm/s f=%0.3f"%(output_time-current_time,fs_temporal,velocity_probe,10*velocity_probe), end="")
#     print(status)
#     velocity_probe += 1
    if len(buffer)<2*frame_count:
        next_frame = True
    if stream_continue:
        return (data, pyaudio.paContinue)
    else:
        return (data, pyaudio.paComplete)

def create_texture(wavelength_texture, length_texture, N_texture, texture_height=0.1):

    fs_spatial = N_texture/(length_texture)

    velocity_probe = 1 # [mm/s]



    if not isinstance(wavelength_texture, list) :
        wavelength_texture = [wavelength_texture]
    
    x = np.linspace(0, length_texture, N_texture, endpoint=False)
    t = x/velocity_probe
    texture = np.zeros(len(x))
    for wavelength in wavelength_texture:
        k_texture = 2*np.pi/wavelength



        texture_height = 0.1
        # texture =  signal.square(k_texture * x)
        texture +=  np.sin(k_texture * x)


    # texture = np.abs(texture)
    texture = np.power(texture,2)

    texture = texture*texture_height

    return (x,texture)


def create_spectrum_texture(wavelength_texture, length_texture, N_texture, velocity_probe=1):
    
    # N_texture = 512*1
    # wavelength_texture = 1/5 # [mm]
    # wavelength_texture = 1/10 # [mm]
    # length_texture = 1
    fs_spatial = N_texture/(length_texture)
    # print("fs_spatial,",fs_spatial)

    # velocity_probe = 1 # [mm/s]

    x,texture = create_texture(wavelength_texture, length_texture, N_texture, texture_height=0.1)

    wavelength_probe = np.inf # [mm]
    # length_probe = 5
    # k_probe = 2*np.pi/wavelength_probe
    # probe = np.cos(k_probe*x)
    # # probe = np.ones(len(x))

    # for a given probe velocity, the spatial frequencies k convert to temporal frequencies as w = k*v

    texture_AC = texture-np.mean(texture)


    # omega_probe = x*k_texture*velocity_probe
    # frequency_probe = x*k_texture/(2*np.pi)*velocity_probe
    spectrum_texture = np.fft.fft(texture_AC)
    fs_temporal = fs_spatial*velocity_probe
    # print("fs_temporal", fs_temporal, "1/t", 1/np.diff(t)[0])

    f = np.fft.fftfreq(spectrum_texture.size, d=1/fs_spatial)
    ft = np.fft.fftfreq(spectrum_texture.size, d=1/fs_temporal)

    spectrum_texture = np.fft.fftshift(spectrum_texture)
    f = np.fft.fftshift(f)

    ft = np.fft.fftshift(ft)

    return (spectrum_texture, fs_spatial)






def easing_impulse_window(a, N):
    
    assert isinstance(a, int), "a must be int"
    
    x = np.arange(a)
    
    A = 0.5-np.cos((x/a)*np.pi)/2
    
    b = N-a
    x = np.arange(b )

    B = 0.5+np.cos((x/(b))*np.pi)/2
    B = -np.power((x/b),2)+1
    
    window = np.concatenate([A,B])
    return window

def texture_easing_window(texture, easing_a, N_easing):
    
#     assert texture.size >= N_easing, "array size must be equal or larger than N_easing"
    if texture.size <= N_easing:
        easing_a = np.int(easing_a/N_easing*texture.size)
        N_easing = texture.size
    
    window = easing_impulse_window(easing_a, N_easing)
    texture = texture[:window.size]*window
    return texture
    
    
    

def add_signal_buffer(buffer_audio, sig):

    # pad the existing buffer to fit the new data
    N_signal = sig.size
    if N_signal > buffer_audio.size:
        # print("a", buffer_audio.size,N_signal)

        buffer_audio = np.pad(buffer_audio, (0, N_signal-buffer_audio.size))
    else: 
        # print( buffer_audio.size,N_signal)
        sig = np.pad(sig, (0, buffer_audio.size-N_signal))
        
        
    buffer_audio = buffer_audio+sig
    return buffer_audio

def extract_signal_buffer(buffer_audio, N_frame):
    
#     assert buffer_audio.size > N_frame, "not enough samples in buffer_audio?"
    if buffer_audio.size < N_frame:
        N_frame = buffer_audio.size
    output = buffer_audio[:N_frame]
    buffer_audio = buffer_audio[N_frame:]
    return output, buffer_audio
    





def resample_texture(texture, k_spatial, fs, velocity, N_useful=None):
    
    # assert velocity>0, "velocity must be non-zero, otherwise just add zeros"

    if velocity == 0:
        # return a zero-ed texture as result
        return np.ones(texture.size)
        
    N_texture = texture.size
    
    
#     if velocity < fs/k_spatial:
    N_resample = np.int(fs/(k_spatial*velocity)*N_texture)
    
    if N_useful is not None:
        # trim texture to useful
#         print("trimmed texture due to low velocity")
        N_texture = np.ceil(k_spatial*velocity/fs*N_useful).astype(np.int)
        texture = texture[:N_texture]
        N_texture = texture.size
    N_resample = np.int(fs/(k_spatial*velocity)*N_texture)
#     print("Ntext", N_texture, "Nresamp", N_resample)
    
#     if N_resample > N_texture:
        
        
    texture_resample = signal.resample(texture, N_resample)
    return texture_resample


def convolve_texture_sample(texture_frame, impulses):
    

    calculated_signal = np.convolve(impulses, texture_frame, mode='full')

    
    return calculated_signal


def generate_impulse_train(velocity, N_frame, fs, delay=0):
    """
    Generates an impulse train respective of the velocity, in frame of size N_frame, sampled at fs
    counting a delay number of samples (float for average finger_velocity) since the last impulse.
    """
    n_impulse_velocity_period = (1/velocity)*fs

    idx = np.array([])

    idx = np.arange(-delay, N_frame, n_impulse_velocity_period)

    
    idx = idx[np.where(idx>=0)]

    if idx.size == 0:
        delay = delay+N_frame
        impulse = np.zeros(N_frame)
    else:
        delay = N_frame-idx[-1]
        idx = np.array(idx).astype(np.int)
#         idx = tuple(idx)
        
        impulse = signal.unit_impulse(N_frame, [i for i in idx])
    
    return (impulse, delay)    
    
    

    













if __name__ == "__main__":

    # Streaming audio buffer to output.
    # With callback function

    # fs_audio = np.int(fs_temporal)
    # print(fs_temporal)
    # N = texture.size


    # creating analytical spectrum texture
    N_texture = 512*1
    wavelength_texture = 1/10 # [mm]
    length_texture = 1

    spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
    


    
    N_audio_segment = 2048# How big is the audio segment size
    N_overlap = 128



   

    N_CHUNK = 512*1
    fs_audio = 8192
    # ratio_fs = fs_audio/fs_temporal

    velocity_probe = 1
    N_output = N_CHUNK
    start_phase = 0

    cutoff = 800



    frame_list = []
    #Initialize
    # sig = generate_audio_from_spectrum(spectrum_texture, N_output=N_output)
    sig = np.zeros(N_CHUNK)
    frequency = 0
    n_phase = 0
    filter_order = 15

    b,a = butter_lowpass_coefficients(cutoff, fs_audio,order=filter_order)
    zf = signal.lfiltic(b,a,sig)

    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = fs_audio
    CHUNK = spectrum_texture.size


    cursor = Cursor()

    # Opening the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    stream_callback=callback,
                    frames_per_buffer=N_CHUNK)



    print("opened")

    stream_continue = 0

    # start the stream (4)
    stream.start_stream()

    stream_continue = 1
    next_frame = True


    velocity_probe = cursor.speed()/500 +10


    buffer = CircularBuffer(maxlen=10*fs_audio)

    time_texture_sample_period = 0.05

    for i in range(1):
        append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )


    N_audio_segment = 1024 # How big is the audio segment size
    N_overlap = 256
    N_audio_segment = N_audio_segment+N_overlap

    time_texture_sample_period = N_audio_segment/fs_audio
    prev_time = time.time()
    while stream.is_active():
        
        
    #     if len(buffer) < N_audio_segment:
        if next_frame:
            next_frame = 0
            velocity_probe = cursor.speed()/500 +10
            fs_temporal = fs_spatial*velocity_probe
            print("\rVelocity %0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal), end="")
            append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
    #         stream_continue = 0
    #         print("Buffer %d is less than the frame count %d"%(len(buffer), N_CHUNK))
    #         break

    #     for i in range(20):
    #     if 1:
    #         curr_time = time.time()
    #         if (curr_time - prev_time > time_texture_sample_period) or next_frame:
    #             print("appending to buffer")
    #             velocity_probe = cursor.speed()/500 +10

    #             print("\r %0.3fs  %0.3fmm/s f=%0.3f"%(curr_time,velocity_probe,10*velocity_probe), end="")
    #             append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
    #             prev_time = curr_time
    #             next_frame = False

                


    print("end")

    stream.stop_stream()
    stream.close()

    p.terminate