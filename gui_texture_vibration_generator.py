import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import wave
import pyaudio
import time


import threading

import win32gui
import win32api


from texture_vibration_generator import *




# the global buffer variable must be defined here.. so callback must be in this file
def callback(in_data, frame_count, time_info, status):

    global zf
    global b,a
    global buffer
    global next_frame
    global window
    global velocity_probe
    global stream_continue
    
    buffer = CircularBuffer(maxlen=10*fs_audio)

    if 1:
        next_frame = 0
        velocity_probe = cursor.speed()/500 +10
        fs_temporal = fs_spatial*velocity_probe
        
        text = "%0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal)
        # print(text)
        window['text_velocity'].update(text)
        append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
        
     
    
    if len(buffer) < 2*frame_count:
        stream_continue = False

    frame = np.copy(buffer.extractleft(frame_count))

#         sig = butter_lowpass_filter(sig, cutoff, fs_audio, order=5)
    frame, zf = signal.lfilter(b, a, frame, zi=zf)
    
    
    audio = array2audio(frame.real)
    data = audio
    
    # output_time = time_info['output_buffer_dac_time']
    # current_time = time_info['current_time']
#     print(N_ratio, sampling_ratio)
#     print("\r %0.3f    fs=%0.3fHz    %0.3fmm/s f=%0.3f"%(output_time-current_time,fs_temporal,velocity_probe,10*velocity_probe), end="")
#     print(status)
# #     velocity_probe += 1
#     if len(buffer)<2*frame_count:
#         next_frame = True

    if stream_continue:
        return (data, pyaudio.paContinue)
    else:
        return (data, pyaudio.paComplete)



def thread_PlayTexture(texture):
    global zf
    global b,a
    global buffer
    global next_frame
    global window
    global velocity_probe
    global stream_continue
    
    # Streaming audio buffer to output.
    print("opened")


    spectrum_texture = texture

    N_audio_segment = 512# How big is the audio segment size
    N_overlap = 128
    N_CHUNK = N_audio_segment
    
    fs_audio = 8192
    # ratio_fs = fs_audio/fs_temporal

    velocity_probe = 1
    # N_output = N_CHUNK
    start_phase = 0

    cutoff = 800



    frame_list = []
    #Initialize
    sig = generate_audio_from_spectrum(spectrum_texture)
    # sig = np.zeros(N_CHUNK)
    # frequency = 0
    # n_phase = 0
    filter_order = 15

    b,a = butter_lowpass_coefficients(cutoff, fs_audio,order=filter_order)
    zf = signal.lfiltic(b,a,sig)

    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = fs_audio
    CHUNK = spectrum_texture.size



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

    for i in range(0):
        append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )


    N_audio_segment = 1024 # How big is the audio segment size
    N_overlap = 256
    N_audio_segment = N_audio_segment+N_overlap

    time_texture_sample_period = N_audio_segment/fs_audio
    prev_time = time.time()


    while stream.is_active():
        
        
        text = " %0.3f mm/s fs_temporal "%(velocity_probe, )
        # print(text)
        window['text_velocity'].update(text)
        # if next_frame and 0:
        #     next_frame = 0
        #     velocity_probe = cursor.speed()/500 +10
        #     fs_temporal = fs_spatial*velocity_probe
            
        #     # window['text_velocity'].update("Velocity %0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal))
        #     append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
        time.sleep(0.1)


    stream.stop_stream()
    stream.close()

    p.terminate





if __name__ == "__main__":
    import matplotlib
    import PySimpleGUI as sg

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    matplotlib.use("TkAgg")

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="left", fill="both", expand=1)
        return figure_canvas_agg


    # # figure
    # fig = plt.figure()

    # t = np.linspace(0,2,1000)
    # y = np.sin(2*np.pi*10*t)

    # plt.plot(t,y)



    
# ------ Menu Definition ------ #      
    menu_def = [['File', ['Exit']],            
            ['Help', 'About...'], ]      

    column1 = [
        [sg.Text("Plot test")],
        [sg.Canvas(key="-CANVAS_TEXTURE_PLOT-", size=(320, 240)) ],
        [sg.Checkbox('Measure Velocity', default=True), sg.Text("Velocity", key='text_velocity')],
        [sg.Button("Play")]
    ]

    column2 = [
        [sg.Text("Sound plot")]
    ]

    # Define the window layout
    layout = [
        [sg.Menu(menu_def, tearoff=True)],  
        [sg.Column(column1), sg.Column(column2)],
        [sg.OK(), sg.Cancel()]
    ]

    # Create the form and show it without the plot
    title_window = "Texture generator"
    window = sg.Window(
        title_window,
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center",
    )

    # Add the plot to the window
    # draw_figure(window["canvas_texture_plot"].TKCanvas, fig)
    # draw_figure(window["canvas_sound"].TKCanvas, fig)

    window["-CANVAS_TEXTURE_PLOT-"].set_tooltip('Texture spectrum')

    


    # Initializing spectrum texture
        # creating analytical spectrum texture
    if 1:
        N_texture = 512*1
        wavelength_texture = 1/10 # [mm]
        length_texture = 1

        spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
        
        fig = plt.figure()
        plt.plot(np.abs(spectrum_texture))


        draw_figure(window["-CANVAS_TEXTURE_PLOT-"].TKCanvas, fig)

    # Initializing Cursor
    if 1:
        cursor = Cursor()


    next_frame = 1
    # Initializing Audio

    window['text_velocity'].update("alksfjhaijhasjk")

    threads = []

    # Event Loop to process "events"
    while True:             
        event, values = window.read(timeout=1000)

        # if event == sg.WIN_CLOSED or event == 'Exit':
        if event in ('Exit', None, 'Cancel'):
            break

        if event in ('Play'):
            # start playing
            print("Playing")    
            # if len(threads) == 0:
            x = threading.Thread(target=thread_PlayTexture, args=(spectrum_texture,))
            threads.append(x)
            x.start()
            pass



    window.close()
