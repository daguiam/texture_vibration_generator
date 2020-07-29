import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import wave
import pyaudio
import time


import threading
import logging

import win32gui
import win32api


from texture_vibration_generator import *






def thread_PlayTexture_blocking():

    global window
    global velocity_probe

    global flag_play_texture
    global spectrum_texture
    
    # Streaming audio buffer to output.
    logging.info("Thread  : Starting")

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

    buffer = CircularBuffer(maxlen=10*fs_audio)


    # Opening the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,)




    N_audio_segment = 1024 # How big is the audio segment size
    N_audio_segment = 2048
    N_overlap = 256
    N_audio_segment = N_audio_segment+N_overlap

    time_texture_sample_period = N_audio_segment/fs_audio
    prev_time = time.time()


    while (flag_play_texture):
        # velocity_probe = cursor.speed()/500 +10
        # gets velocity from global variable.
        # starts new data
        append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
        
        frame = np.copy(buffer.extractleft(N_audio_segment))
        frame, zf = signal.lfilter(b, a, frame, zi=zf)
        audio = array2audio(frame.real)
        data = audio

        stream.write(data)



    stream.stop_stream()
    stream.close()

    p.terminate
    
    logging.info("Thread  : Stopping")



if __name__ == "__main__":
    import matplotlib
    import PySimpleGUI as sg

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    matplotlib.use("TkAgg")

    global flag_play_texture

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="left",  expand=1)
        return figure_canvas_agg

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : Started program")
    # # figure
    # fig = plt.figure()

    # t = np.linspace(0,2,1000)
    # y = np.sin(2*np.pi*10*t)

    # plt.plot(t,y)



    sg.theme('SystemDefaultForReal') 
    # sg.theme('SystemDefault')
    # sg.theme('SystemDefault1') 
 

# ------ Menu Definition ------ #      
    menu_def = [['File', ['Exit']],            
            ['Help', 'About...'], ]      

    column1 = [
        [sg.Text("Surface texture and spectrum")],
        [sg.Listbox(values=('Synthetic 1', 'Sample A', 'Sample B'), size=(30, 6), key='listbox_texture_input'),],
        [sg.Canvas(key="-CANVAS_TEXTURE_PLOT-", size=(320, 240)) ],
        
    ]

    column2 = [
        [sg.Text("Sound plot")],
        [sg.Checkbox('Measure Velocity', default=True), sg.Text("Velocity mm/s", key='text_velocity')],
        [sg.Canvas(key="-CANVAS_VELOCITY_PLOT-", size=(100, 50)) ],

        [sg.Button("Play", key='btn_play'),sg.Button("Stop", key='btn_stop')]
    ]

    # Define the window layout
    layout = [
        [sg.Menu(menu_def, tearoff=True)],  
        [sg.Column(column1),sg.VerticalSeparator(), sg.Column(column2)],
        # [sg.OK(), sg.Cancel()]
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

    


    # Initializing spectrum texture
        # creating analytical spectrum texture
    if 1:
        N_texture = 512*1
        wavelength_texture = 1/10 # [mm]
        length_texture = 1
        x, texture = create_texture(wavelength_texture, length_texture, N_texture)
        spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
        
        figure_texture,ax = plt.subplots(2,1, figsize=(4,4))
        plt.sca(ax[0])
        plt.plot(x,texture)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.sca(ax[1])
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(spectrum_texture))),np.abs(spectrum_texture))
        plt.xlabel('k [1/mm]')
        plt.ylabel('A [a.u.]')

        plt.xlim(xmin=0)
        plt.subplots_adjust(hspace=0.4, left=0.15)

        window["-CANVAS_TEXTURE_PLOT-"].set_tooltip('Texture spectrum')
        agg_texture = draw_figure(window["-CANVAS_TEXTURE_PLOT-"].TKCanvas, figure_texture)

    # Initializing Cursor
    if 1:
        cursor = Cursor()

        max_samples_velocity = 200
        buffer_velocity = CircularBuffer(maxlen=max_samples_velocity)
        buffer_velocity_t = CircularBuffer(maxlen=max_samples_velocity)

        velocity_probe = cursor.speed()

        buffer_velocity.extend([velocity_probe])
        buffer_velocity_t.extend([time.time()])
        figure_velocity,ax = plt.subplots(figsize=(4,2))


        plt.plot(np.array(buffer_velocity_t)-np.array(buffer_velocity_t)[0], buffer_velocity)
        
        plt.xlabel('Time')
        plt.ylabel('Velocity [mm/s]')
        plt.subplots_adjust( bottom=0.25, left=0.15)

        window["-CANVAS_TEXTURE_PLOT-"].set_tooltip('Velocity plot')
        agg_velocity = draw_figure(window["-CANVAS_VELOCITY_PLOT-"].TKCanvas, figure_velocity)


    next_frame = 1
    # Initializing Audio

    texture = spectrum_texture

    threads_texture = list()

    # Event Loop to process "events"
    while True:             
        event, values = window.read(timeout=10)
        velocity_probe = cursor.speed()/500 +10
        window['text_velocity'].update("V=%0.1fmm/s"%(velocity_probe))

        # add to velocity figure
        if 1:
            buffer_velocity.extend([velocity_probe])
            buffer_velocity_t.extend([time.time()])
            ax = figure_velocity.axes[0]
            ax.cla()
            ax.plot(np.array(buffer_velocity_t)-np.array(buffer_velocity_t)[-1], buffer_velocity)
            ax.set_xlim(-20,0)
            
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Velocity [mm/s]')
            # plt.subplots_adjust( bottom=0.6, left=0.15)
            agg_velocity.draw()


        # if event == sg.WIN_CLOSED or event == 'Exit':
        if event in ('Exit', None, 'Cancel'):
            logging.info("Main    : Exiting")

            break

        
        print("akjsdhjkas", values['listbox_texture_input'], values)


        if event in ('btn_play'):
            logging.info("Main    : Play")
           # start playing
            # if len(threads) == 0:
            if len(threads_texture):
                flag_play_texture = False
                for thread in threads_texture:
                    thread.join()
                    threads_texture.remove(thread)

            flag_play_texture = True
            x = threading.Thread(target=thread_PlayTexture_blocking, args=(), daemon=True)
            threads_texture.append(x)
            
            x.start()
            
        if event in ('btn_stop'):
            logging.info("Main    : Stop")

            flag_play_texture = False
    window.close()
