import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import wave
import pyaudio
import time



import win32gui



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
    sig = np.fft.ifft(spectrum)
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





def callback(in_data, frame_count, time_info, status):
    global velocity_probe
    global sig
    global zf
    global b,a
    global frequency
    global n_phase
    N_output = frame_count
   
    velocity_probe = cursor.speed()/500 +1

    
    if velocity_probe ==0 :
        sig_frame = np.zeros(N_CHUNK)
    else:
    #     print(N_CHUNK,fs_audio,fs_spatial, N_output)
        N_spectrum = N_output = spectrum_texture.size

    #     velocity_probe = 3
        N_resampled = np.int(velocity_probe*fs_spatial/fs_spatial*N_spectrum)
    #     print(N_resampled)



        # first get signal from texture sampled at fs_spatial
        N_texture = len(spectrum_texture)
        sig_spatial = generate_audio_from_spectrum(spectrum_texture, N_output=None)
#         reps = 2
#         sig_spatial = np.tile(sig_spatial, reps)
        N_spatial = len(sig_spatial)
        t_spatial = np.linspace(0, N_texture/fs_spatial, N_spatial)


        # resample the signal to the fs_temporal = fs_spatial*velocity_probe

        fs_temporal = fs_spatial*velocity_probe
        # N_temporal = velocity_probe*N_spatial
        reps = 5

        sig_temporal = np.tile(sig_spatial, reps)
        N_temporal = len(sig_temporal)



    #     t_temporal = np.linspace(0, N_temporal/fs_temporal, N_temporal)



        # CHUNK timespan:

        t_CHUNK = N_CHUNK/fs_audio
    #     print(t_CHUNK)
        # t_audio = np.linspace(0,t_CHUNK, N_CHUNK)

        N_audio = np.int(fs_audio/fs_temporal*N_temporal)

        #resample to the sampling frequency fs_audio
        sig_audio = signal.resample(sig_temporal, N_audio)
        t_audio = np.linspace(0,N_audio/fs_audio, N_audio)

    #     n_phase = 0

        t_CHUNK = N_CHUNK/fs_audio
        sig_frame = sig_audio[n_phase:np.mod(N_CHUNK+n_phase, N_audio)]
        t_frame = np.linspace(0,t_CHUNK, N_CHUNK)
#         n_phase = np.mod(N_CHUNK+n_phase, N_audio)

#     sig = np.sin(phase)
#     prev_phase = phase[-1]
        
        
    
    frame = np.copy(sig_frame)

#         sig = butter_lowpass_filter(sig, cutoff, fs_audio, order=5)
    frame, zf = signal.lfilter(b, a, frame, zi=zf)
    
    audio = array2audio(frame.real)
    data = audio
#     print(frame_count,data.size, N_ratio, sampling_ratio, fs_temporal, fs_audio, sampling_ratio*fs_audio)

#     print("\r%0.3f %0.3f %0.3f"%(velocity_probe, fs_spatial, fs_temporal), end="")
#     print(time_info)
    
    output_time = time_info['output_buffer_dac_time']
    current_time = time_info['current_time']
#     print(N_ratio, sampling_ratio)
    print("\r %0.3f    %0.3f    %0.3f"%(output_time-current_time,fs_temporal,velocity_probe), end="")
#     print(status)
#     velocity_probe += 1
    if stream_continue:
        return (data, pyaudio.paContinue)
    else:
        return (data, pyaudio.paComplete)

if __name__ == "__main__":

    # Streaming audio buffer to output.
    # With callback function




    N_CHUNK = 512*2
    fs_audio = 8192
    # ratio_fs = fs_audio/fs_temporal

    velocity_probe = 1
    N_output = N_CHUNK
    start_phase = 0

    cutoff = 512

    sig = np.zeros(N_CHUNK)

        

        
    #Initialize
    # sig = generate_audio_from_spectrum(spectrum_texture, N_output=N_output)
    sig = np.zeros(N_CHUNK)
    frequency = 0

    b,a = butter_lowpass_coefficients(cutoff, fs_audio, order=15)
    # zf = signal.lfiltic(b,a,sig)
    zi = signal.lfilter_zi(b, a)

    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = fs_audio
    # CHUNK = spectrum_texture.size


    cursor = Cursor()

    # Opening the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    stream_callback=callback,
                    frames_per_buffer=N_CHUNK,)
                    # output_device_index=5)



    print("opened")



    # start the stream (4)
    stream.start_stream()

    stream_continue = 1



    # wait for stream to finish (5)


    # @interact(velocity=1)
    # def get_velocity_probe(velocity):
    #     global velocity_probe
    #     velocity_probe = velocity
    #     return velocity
    try:
        while stream.is_active():
            time.sleep(0.05)
        #     print(velocity_probe)
            velocity_probe = 1
    #         velocity_probe = cursor.speed()/500+1
    #         velocity_probe = get_velocity_probe()


            velocity_probe = cursor.speed()/500+1
            # t = np.linspace(0,N_CHUNK/fs_audio,N_CHUNK)
            # frequency = 100*velocity_probe
            # frequency = np.floor(frequency)
            # sig = np.sin(2*np.pi*frequency*t)
        
            print("\r %04.3f    %04.3f"%(frequency, np.ptp(sig)), end="")

            # print("pk-pk",np.ptp(sig))


    except KeyboardInterrupt:
        pass


        print("end")

        stream.stop_stream()
        stream.close()

        p.terminate


    print("end")
