import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import wave
import pyaudio
import time

import json
import os


import threading
import logging

import win32gui
import win32api


from texture_vibration_generator import *






def thread_PlayTexture_blocking(output_device_index=None):

    global window
    global velocity_probe

    global flag_play_texture
    global spectrum_texture
    
    # Streaming audio buffer to output.
    logging.info("Thread  : Starting")

    # N_audio_segment = 512# How big is the audio segment size
    # N_overlap = 128
    # N_CHUNK = N_audio_segment
    
    fs_audio = 8192
    # fs_audio = 4096*4
    # ratio_fs = fs_audio/fs_temporal

    # velocity_probe = 1
    # N_output = N_CHUNK
    start_phase = 0

    # cutoff = 400



    frame_list = []
    #Initialize
    sig = generate_audio_from_spectrum(spectrum_texture)
    sig = np.zeros(sig.size)
    # sig = np.zeros(N_CHUNK)
    # frequency = 0
    # n_phase = 0
    filter_order = 3

    lowcut = 50
    highcut = 400

    b,a = butter_lowpass_coefficients(highcut, fs_audio,order=filter_order)
    b,a = butter_bandpass(lowcut, highcut, fs_audio, order=filter_order)
    # b,a = butter_bandpass(2, 400, fs_audio, order=filter_order)

    zf = signal.lfiltic(b,a, sig)
    
    # print("zf",zf)

    p = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = fs_audio
    # CHUNK = spectrum_texture.size

    buffer = CircularBuffer(maxlen=10*fs_audio)
    # buffer = CircularBuffer(maxlen=2048)


    # Opening the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer = 2048*2,
                    output_device_index=output_device_index,)




    N_audio_segment = 1024 # How big is the audio segment size
    N_audio_segment = 512*2
    N_overlap = 256*0
    # N_audio_segment = N_audio_segment+N_overlap

    time_texture_sample_period = N_audio_segment/fs_audio
    prev_time = time.time()
    prev_time2 = time.time()

    data = np.zeros(N_audio_segment)
    while (flag_play_texture):
        # velocity_probe = cursor.speed()/500 +10
        # gets velocity from global variable.
        # starts new data

        stream.write(data)


        # velocity_probe += 10
        # velocity_probe = 15
        
        # removed circular buffer.. calculates new texture frame every time.
        if 0:
            added_size = append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
            logging.info("Size of buffer: %d, added %d"%(len(buffer),added_size))


            frame = np.copy(buffer.extractleft(N_audio_segment))
            frame = np.copy(buffer.extract(added_size))

        # t_frame, sig_frame = estimate_texture_signal(spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, )

        t_frame = np.linspace(0, N_audio_segment/fs_audio, N_audio_segment)
        sig_frame = np.sin(2*np.pi*256*t_frame)
        # print(sig_frame)


        frame = sig_frame

        logging.info("Frame size: %d  frame diff %f ptp %f"%(frame.size, frame[-1]-frame[0], np.ptp(frame)))
        # print(np.diff(t_frame)[:5])
        
        frames_available = stream.get_write_available()
        output_latency = stream.get_output_latency()

        logging.info("Write available %d, latency %f"%(frames_available,output_latency))


        # logging.info("Buffer size before %d"%(len(buffer)))
        # frame = np.copy(buffer.extractleft(frames_available))

       
        # zf = None
        # logging.info("Buffer size %d"%(len(buffer)))
        # print("ptp", np.ptp(frame))
        # print("ptp", np.ptp(frame), np.mean(frame))

        # print("Frame max:", np.max(frame.real))

        # frame, zf = signal.lfilter(b, a, frame, zi=zf)
        
        # frame = signal.lfilter(b, a, frame, )

        # print("ptpfilt", np.ptp(frame), np.mean(frame))

        # frame *= np.hamming(frame.size)

        # frame = frame*output_volume/100
        # frame = frame/np.max(frame)*output_volume/100
        
        # assert np.max(frame.real)<=1, "Frame overflowed amplitude=1"
        # print("Frame max:", np.max(frame.real))

        if np.max(frame.real)>=1:
            print("Overflowed amplitude: %f"%(np.max(frame.real)))
            logging.info("Overflowed amplitude: %f"%(np.max(frame.real)))
            continue
        audio = array2audio(frame.real, max_amplitude=1)
        data = audio
        # data = np.repeat(data,2)


        # stream.write(data)
        curr_time = time.time()
        # logging.debug("Time delta %0.4f %0.4f and %0.4f s %f"%(curr_time-prev_time,curr_time-prev_time2, N_audio_segment/fs_audio, velocity_probe))

        # logging.info("Write available %d"%(stream.get_write_available()))
        
        prev_time2 = prev_time
        prev_time = curr_time



    stream.stop_stream()
    stream.close()

    p.terminate()
    
    logging.info("Thread  : Stopping")



# Fill list of textures
list_of_textures = ['Texture 1', 'Texture 2', 'Texture 3']
textures_dictionary = dict()
json_filename = os.path.join("textures", "textures.json")
with open(json_filename) as json_file:
    textures_dictionary = json.load(json_file)

textures_gain = 1/100
list_of_textures += sorted(textures_dictionary.keys())


list_of_output_devices = []
list_of_output_devices_selected = None
list_of_output_devices_selected_device_id = None
output_volume = 20

gui_timetout = 10  #ms
gui_plot_velocity_time = 50

base_velocity = 10
max_samples_velocity = 200

DEBUG_VERBOSE = False
# DEBUG_VERBOSE = True


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

    format = "%(asctime)s.%(msecs)03d: %(message)s"
    if DEBUG_VERBOSE:
        logging.basicConfig(format=format, level=logging.DEBUG,
                            datefmt="%H:%M:%S")
    else:
        logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : Started program")
    # # figure
    # fig = plt.figure()

    # t = np.linspace(0,2,1000)
    # y = np.sin(2*np.pi*10*t)

    # plt.plot(t,y)

    # Listing output devices
    if 1:
        logging.info("Main    : Listing output stream devices")
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for device_id in range(num_devices):
            device = p.get_device_info_by_host_api_device_index(0,device_id)
        #     pprint.pprint(device)
            if device.get('maxOutputChannels') > 0:
                name = device.get('name')
                if "Cypress" in name:
                    list_of_output_devices_selected = name
                    list_of_output_devices_selected_device_id = device.get('index')
                list_of_output_devices.append(name)
                
                logging.info("Output devices : [%d] %s"%(device.get('index'),device.get('name'),))
                
        p.terminate()



    sg.theme('SystemDefaultForReal') 
    # sg.theme('SystemDefault')
    # sg.theme('SystemDefault1') 
 

# ------ Menu Definition ------ #      
    menu_def = [['File', ['Exit']],            
            ['Help', 'About...'], ]      

    column1 = [
        [sg.Text("Surface texture and spectrum")],
        [sg.Listbox(values=list_of_textures, default_values=list_of_textures[0], size=(30, 6), key='listbox_texture_input'),],
        [sg.Canvas(key="-CANVAS_TEXTURE_PLOT-", size=(320, 240)) ],
        
    ]

    column2 = [
        [sg.Text("Sound plot")],
        [ sg.Text("Velocity mm/s", key='text_velocity'),
            sg.Checkbox('Add base velocity', default=True, key='checkbox_base_velocity'),
            sg.Slider(range=(0, 30), orientation='h', size=(20, 5), default_value=10, tick_interval=10, key='slider_base_velocity')],
        [sg.Canvas(key="-CANVAS_VELOCITY_PLOT-", size=(100, 50)) ],
        [sg.Text("Select output device [Cypress]"),sg.Button("Refresh", key='btn_refresh_devices')],
        [sg.Listbox(values=list_of_output_devices, default_values=list_of_output_devices_selected, size=(30, 6), key='listbox_output_devices'),
          sg.Column(
              [[sg.Text('Volume')],
            [sg.Slider(range=(0, 100), orientation='v', size=(6, 10), default_value=output_volume, tick_interval=20, key='slider_output_volume')],
            ]), ],
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
        
        figure_texture,ax = plt.subplots(1,1, figsize=(4,2))
        # plt.sca(ax[0])
        # plt.plot(x,texture)
        # plt.xlabel('x [mm]')
        # plt.ylabel('y [mm]')
        
        # plt.sca(ax[1])
        plt.plot(np.fft.fftshift(np.fft.fftfreq(len(spectrum_texture))),np.abs(spectrum_texture))
        plt.xlabel('k [1/mm]')
        plt.ylabel('A [a.u.]')

        plt.xlim(xmin=0)
        plt.subplots_adjust(hspace=0.4, left=0.15, bottom=0.25)

        window["-CANVAS_TEXTURE_PLOT-"].set_tooltip('Texture spectrum')
        agg_texture = draw_figure(window["-CANVAS_TEXTURE_PLOT-"].TKCanvas, figure_texture)

    # Initializing Cursor
    if 1:
        cursor = Cursor()

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

        window["-CANVAS_VELOCITY_PLOT-"].set_tooltip('Velocity plot')
        agg_velocity = draw_figure(window["-CANVAS_VELOCITY_PLOT-"].TKCanvas, figure_velocity)


    next_frame = 1
    # Initializing Audio

    texture = spectrum_texture

    threads_texture = list()

    curr_time_plot = time.time()
    prev_time_plot = curr_time_plot

    # Event Loop to process "events"
    while True:             
        event, values = window.read(timeout=gui_timetout)
        # print("Read events", values['listbox_texture_input'], values)


        velocity_probe = cursor.speed()/200 + base_velocity
        window['text_velocity'].update("V=%0.1fmm/s"%(velocity_probe))

        output_volume = values['slider_output_volume']

        if values['checkbox_base_velocity']:
            base_velocity = values['slider_base_velocity']
        else:
            base_velocity = 0

        # add to velocity figure
        if 1:
            buffer_velocity.extend([velocity_probe])
            buffer_velocity_t.extend([time.time()])
            curr_time_plot = time.time()

            if curr_time_plot - prev_time_plot > gui_plot_velocity_time/1000:
                ax = figure_velocity.axes[0]
                ax.cla()
                ax.plot(np.array(buffer_velocity_t)-np.array(buffer_velocity_t)[-1], buffer_velocity)
                ax.set_xlim(-20,0)
                
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('Velocity [mm/s]')
                # plt.subplots_adjust( bottom=0.6, left=0.15)
                agg_velocity.draw()
                prev_time_plot = curr_time_plot

        # if event == sg.WIN_CLOSED or event == 'Exit':
        if event in ('Exit', None, 'Cancel'):
            logging.info("Main    : Exiting")

            break

        


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
            
            
            # Selecting output device_id from list_box
            if 1:
                selected_output_device = values['listbox_output_devices'][0]

                p = pyaudio.PyAudio()
                info = p.get_host_api_info_by_index(0)
                num_devices = info.get('deviceCount')
                device_index = None
                for device_id in range(num_devices):
                    device = p.get_device_info_by_host_api_device_index(0,device_id)
                #     pprint.pprint(device)
                    if device.get('maxOutputChannels') > 0:
                        name = device.get('name')
                        if selected_output_device == name:
                            device_index = device.get('index')
                        
                            logging.info("Main    : Play output on [%d] %s"%(device.get('index'),device.get('name'),))
                            break
                        
            x = threading.Thread(target=thread_PlayTexture_blocking, args=([device_index]), daemon=True)
            threads_texture.append(x)
            
            x.start()
            
        if event in ('btn_stop'):
            logging.info("Main    : Stop")

            flag_play_texture = False

        if event in ('btn_refresh_devices'):
            logging.info("Main    : Refreshing device list")

            list_of_output_devices = []
            logging.info("Main    : Listing output stream devices")
            p = pyaudio.PyAudio()
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for device_id in range(num_devices):
                device = p.get_device_info_by_host_api_device_index(0,device_id)
            #     pprint.pprint(device)
                if device.get('maxOutputChannels') > 0:
                    name = device.get('name')
                    if "Cypress" in name:
                        list_of_output_devices_selected = name
                        list_of_output_devices_selected_device_id = device.get('index')
                    list_of_output_devices.append(name)
                    
                    logging.info("Output devices : [%d] %s"%(device.get('index'),device.get('name'),))
            window['listbox_output_devices'].update(list_of_output_devices)
            window['listbox_output_devices'].SetValue(list_of_output_devices_selected)
            p.terminate()





            
        # Updating the spectrum texture         
        # creating analytical spectrum texture
        if 1:
            N_texture = 512*1

            selected_texture = values['listbox_texture_input'][0]

            if selected_texture.lower() in ["texture 1", "texture 2", "texture 3"]:
                if selected_texture.lower() in ["texture 1"]:

                    wavelength_texture = 1/10 # [mm]
                elif selected_texture.lower() in ["texture 2"]:
                    wavelength_texture = 1/20 # [mm]
                    wavelength_texture = [1/20, 1/20]

                else:
                    wavelength_texture = 1/30 # [mm]
                
                    
                # print("Wavelength texture", wavelength_texture,selected_texture)
                length_texture = 1
                x, texture = create_texture(wavelength_texture, length_texture, N_texture)
                spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
                
                spectrum_texture = np.abs(spectrum_texture)
                spectrum_texture /= np.max( np.abs(spectrum_texture))
                # spectrum_texture *= output_volume

                axes = figure_texture.axes
                ax = axes[0]
                # ax = axes[0]
                # ax.cla()
                # ax.plot(x,texture)
                # ax.set_xlabel('x [mm]')
                # ax.set_ylabel('y [mm]')
                
                # ax = axes[1]
                
                ax.cla()
                ax.plot(np.fft.fftshift(np.fft.fftfreq(len(spectrum_texture), d=1/fs_spatial)),np.abs(spectrum_texture))
                ax.set_xlabel('k [1/mm]')
                ax.set_ylabel('A [a.u.]')

                ax.set_xlim(xmin=0)
                # plt.subplots_adjust(hspace=0.4, left=0.15)
                agg_texture.draw()
            else:

                texture_file = textures_dictionary[selected_texture]
                freqs, spectrum = np.loadtxt(texture_file, delimiter=',', unpack=True)
                # spectrum = spectrum/np.max(spectrum)
                spectrum_texture = spectrum
                spectrum_texture *= textures_gain
                # spectrum_texture *= output_volume

                

                axes = figure_texture.axes
                # clar first axis
                ax = axes[0]
                ax.cla()


                # ax = axes[1]
                ax.cla()
                ax.plot(freqs, spectrum)
                ax.set_xlabel('k [1/mm]')
                ax.set_ylabel('A [a.u.]')
                # ax.set_xlim(xmin=0, xmax=40)
                ax.set_xlim(xmin=0, xmax=100)
                ax.set_ylim(0,1)

                agg_texture.draw()


    window.close()
