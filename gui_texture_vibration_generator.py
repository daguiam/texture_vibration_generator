# import simpleaudio as sa
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

import libGTVelocityStreamer


IP = "127.0.0.1"
PORT = 28900


def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

from texture_vibration_generator import *



def signal_from_spectrum(spectrum, N_trim=0):

    sig = np.fft.ifft(spectrum)
    N_trim = np.int(N_trim)
    N_sig = sig.size

    if N_sig > 2*N_trim:

        sig = sig[N_trim:N_sig-N_trim]
    return sig


def thread_PlayTexture_blocking(output_device_index=None):

    global window
    global velocity_probe

    global flag_play_texture
    global spectrum_texture
    
    global lowcut, highcut
    # Streaming audio buffer to output.
    logging.info("Thread  : Starting play texture blocking")

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

    sig = np.zeros(fs_audio)
    # sig = np.zeros(N_CHUNK)
    # frequency = 0
    # n_phase = 0
    filter_order = 3

    N_easing = 1024*2
    easing_a = 512

    N_easing = 512*2
    easing_a = 128



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



    N_audio_segment = 1024 # How big is the audio segment size
    N_audio_segment = 128
    N_overlap = 256*0
    # Opening the stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer = N_audio_segment,
                    output_device_index=output_device_index,)



    # fs_texture = 1000
    # N_audio_segment = N_audio_segment+N_overlap

    time_texture_sample_period = N_audio_segment/fs_audio
    prev_time = time.time()
    prev_time2 = time.time()

    data = np.zeros(N_audio_segment)
    delay = 0
    buffer_audio = np.array([])
    output_buffer = np.array([])

    output_latency = stream.get_output_latency()
    logging.info("Thread  : Output Latency %0.3fs"%(output_latency))

    while (flag_play_texture):
        # velocity_probe = cursor.speed()/500 +10
        # gets velocity from global variable.
        # starts new data

        stream.write(data, data.size)

        N_texture = spectrum_texture.size

        # velocity_probe += 10
        # velocity_probe = 15
        
        

        if 1:
            

            # velocity for each frame
            N_frame = N_audio_segment

            if velocity_probe == 0:
                calculated_signal = np.zeros(N_frame)
            else:
                texture_frame = resample_texture(spectrum_texture, fs_spatial, fs_audio, velocity_probe, N_useful=N_easing)

                if 1:
                    transient_x = np.linspace(0,len(texture_frame)/fs_audio, len(texture_frame))
                    texture_frame = texture_frame*1e-6
                    # print(len(transient_x),len(texture_frame))
                    d_texture = np.gradient(texture_frame, transient_x)
                    d2_texture = np.gradient(d_texture, transient_x)
                    # d2_texture = d2_texture/np.max(d2_texture)
                    # logging.info("Size of d2 %f"%(np.ptp(d2_texture)))

                    texture_frame = d2_texture

                finger_velocity = velocity_probe*finger_spacing
                impulses, delay = generate_impulse_train(finger_velocity, N_frame, fs_audio, delay)

                # easing the impulse signal
                texture_frame = texture_easing_window(texture_frame, easing_a, N_easing)

                # texture_frame = texture_frame/np.max(np.abs(texture_frame))
                calculated_signal = convolve_texture_sample(texture_frame, impulses)

            buffer_audio = add_signal_buffer(buffer_audio, calculated_signal)           
            output, buffer_audio = extract_signal_buffer(buffer_audio, N_frame)
            
            # output_buffer = np.concatenate([output_buffer, output])
            sig_frame = output
            

        
        frame = sig_frame
        frame = frame*output_volume/100

        # print('ptp frame', np.ptp(frame), len(buffer_audio), sig_frame.size)

        # logging.info("Frame size: %d  frame diff %f ptp %f, fsspatial %d  Naudio %d Ntext %d"%(
        #     frame.size, frame[-1]-frame[0], np.ptp(frame), fs_spatial, N_audio_segment, N_texture))
        
        
        
        frames_available = stream.get_write_available()
        output_latency = stream.get_output_latency()

        # logging.info("Write available %d, latency %f"%(frames_available,output_latency))


        # logging.info("Buffer size before %d"%(len(buffer)))
        # frame = np.copy(buffer.extractleft(frames_available))

       

        frame, zf = signal.lfilter(b, a, frame, zi=zf)
        
        # print("frame amplitude: %f std %f frame start %f end %f"%(np.ptp(frame.real), np.std(frame.real), frame.real[0], frame.real[-1]))



        if np.max(frame.real)>=1:
            print("Overflowed amplitude: %f"%(np.max(frame.real)))
            logging.info("Overflowed amplitude: %f"%(np.max(frame.real)))
            # continue
        audio = array2audio(frame.real, max_amplitude=2)
        data = audio
        # data = np.repeat(data,2)


        # stream.write(data)
        curr_time = time.time()
        logging.debug("Time delta %3.4f %3.4f and %3.4f s %3.4f Write available %d ptp %3.4f"%(curr_time-prev_time,curr_time-prev_time2, N_audio_segment/fs_audio, velocity_probe,stream.get_write_available(), np.ptp(frame.real)))

        # logging.info("Write available %d"%(stream.get_write_available()))
        
        prev_time2 = prev_time
        prev_time = curr_time



    stream.stop_stream()
    stream.close()

    p.terminate()
    
    logging.info("Thread  : Stopping")










# Finger details
finger_spacing = 1

# frequency response
lowcut = 50
highcut = 600
    

# Fill list of textures
list_of_textures = ['Texture 1', 'Texture 2', 'Texture 3']
list_of_textures = [] # synthetic textures removed
textures_dictionary = dict()
json_filename = os.path.join("textures", "textures.json")
with open(json_filename) as json_file:
    textures_dictionary = json.load(json_file)

textures_gain = 0.1
list_of_textures += sorted(textures_dictionary.keys())
selected_texture = list_of_textures[0]


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
                list_of_output_devices_selected = name
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
        [sg.Listbox(values=list_of_textures, default_values=selected_texture, size=(30, 6), key='listbox_texture_input'),],
        [sg.Text("Textures gain: "),sg.Input(textures_gain, key='input_textures_gain',size=(10,1), enable_events=True),
         sg.Button("Set", key='btn_set_textures_gain'), sg.Text("", key='warning_textures_gain_overflow')],
        [sg.Canvas(key="-CANVAS_TEXTURE_PLOT-", size=(320, 240)) ],
        
    ]
    column2 = [
        [sg.Text("Vibration Synhesizer")],
        [sg.Frame(layout=[ 
            [sg.Radio('Geomagic Touch', "radio_velocity_source", default=True, key='radio_velocity_source_geomagic')],
            [sg.Radio('Mouse cursor', "radio_velocity_source", key='radio_velocity_source_mouse')],
            [sg.Checkbox('Add base velocity', default=False, key='checkbox_base_velocity'),
              sg.Slider(range=(0, 60), orientation='h', size=(20, 5), default_value=10, tick_interval=10, key='slider_base_velocity')],
        
            [ sg.Text("Velocity mm/s", key='text_velocity'),],
            [sg.Canvas(key="-CANVAS_VELOCITY_PLOT-", size=(100, 50)) ],
        ],title='Velocity')],
        
        [sg.Text("Select output device [Cypress]"),sg.Button("Refresh", key='btn_refresh_devices')],
        [sg.Listbox(values=list_of_output_devices, default_values=list_of_output_devices_selected, size=(30, 6), key='listbox_output_devices'),
          sg.Column(
              [[sg.Text('Volume')],
            [sg.Slider(range=(0, 100), orientation='v', size=(6, 10), default_value=output_volume, tick_interval=20, key='slider_output_volume')],
            ]), ],
        [sg.Text("Audio filter")],
        [sg.Text("Low: "),sg.Input(lowcut, key='input_filter_lowcutoff',size=(5,1), enable_events=True),sg.Text("Hz"),
            sg.Text("High: "),sg.Input(highcut, key='input_filter_highcutoff',size=(5,1), enable_events=True),sg.Text("Hz"),
             sg.Button("Set", key='btn_set_filter') ],
        [sg.Button("Play", key='btn_play'),sg.Button("Stop", key='btn_stop'), 
            sg.StatusBar( text=f'Stopped', size=(10,1), justification='left', visible=True, key='play_status_bar' )]
    ]

    # Define the window layout
    layout = [
        [sg.Menu(menu_def, tearoff=True)],  
        [sg.Column(column1),sg.VerticalSeparator(), sg.Column(column2)],
        [sg.StatusBar( text=f'', size=(50,1), justification='left', visible=True, key='status_bar' )],
 
        # [sg.OK(), sg.Cancel()]
    ]


    # icon_base64 = 
    # Create the form and show it without the plot
    title_window = "Vibrotactile virtual texture generator"
    window = sg.Window(
        title_window,
        layout,
        location=(0, 0),
        finalize=True,
        element_justification="center",
        icon = sg.DEFAULT_BASE64_ICON,

    )
    window['status_bar'].expand(expand_x=True, expand_y=True)

    # Add the plot to the window
    # draw_figure(window["canvas_texture_plot"].TKCanvas, fig)
    # draw_figure(window["canvas_sound"].TKCanvas, fig)

    


    # Initializing spectrum texture
        # creating analytical spectrum texture
    if 1:
        # N_texture = 512*1
        # wavelength_texture = 1/10 # [mm]
        # length_texture = 1
        # x, texture = create_texture(wavelength_texture, length_texture, N_texture)
        # spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
        
        texture_file = textures_dictionary[selected_texture]
        freqs, spectrum = np.loadtxt(texture_file, delimiter=',', unpack=True)
        fs_spatial = np.int(np.ptp(freqs)+1)
        # print("fs_spatial",fs_spatial)
        # spectrum = spectrum/np.max(spectrum)



        spectrum_texture = spectrum
        spectrum_texture *= textures_gain

        figure_texture,ax = plt.subplots(1,1, figsize=(4,2))
        # plt.sca(ax[0])
        # plt.plot(x,texture)
        # plt.xlabel('x [mm]')
        # plt.ylabel('y [mm]')
        
        # plt.sca(ax[1])
        # plt.plot(np.fft.fftshift(np.fft.fftfreq(len(spectrum_texture))),np.abs(spectrum_texture))
        plt.plot(freqs, spectrum)

        plt.xlabel('k [1/mm]')
        plt.ylabel('A [a.u.]')

        plt.xlim(xmin=0)
        plt.subplots_adjust(hspace=0.4, left=0.20, bottom=0.25)

        window["-CANVAS_TEXTURE_PLOT-"].set_tooltip('Texture spectrum')
        agg_texture = draw_figure(window["-CANVAS_TEXTURE_PLOT-"].TKCanvas, figure_texture)

    # Initializing Cursor
    if 1:
        cursor = Cursor()

        gt_device = libGTVelocityStreamer.GTVelocityStreamer(ip=IP, port=PORT,)

        # while not ~gt_device.check_alive():
        #     print("Not alive",gt_device.check_alive())
        #     gt_device.close()
        #     gt_device = libGTVelocityStreamer.GTVelocityStreamer(ip=IP, port=PORT,)
        #     time.sleep(0.1)


        buffer_velocity = CircularBuffer(maxlen=max_samples_velocity)
        buffer_velocity_t = CircularBuffer(maxlen=max_samples_velocity)

        velocity_probe = cursor.speed()
        velocity_probe = gt_device.speed()

        



        buffer_velocity.extend([velocity_probe])
        buffer_velocity_t.extend([time.time()])
        figure_velocity,ax = plt.subplots(figsize=(4,2))


        plt.plot(np.array(buffer_velocity_t)-np.array(buffer_velocity_t)[0], buffer_velocity)
        
        plt.xlabel('Time')
        plt.ylabel('Velocity [mm/s]')
        plt.subplots_adjust( bottom=0.25, left=0.20)

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



        if values['radio_velocity_source_mouse']:
            velocity_probe = cursor.speed()/200 + base_velocity
        elif values['radio_velocity_source_geomagic']:
            velocity_probe = gt_device.speed() + base_velocity
            if not gt_device.alive:
                window['status_bar'].update("Warning: GTVelocityStreamer not alive")
            else:
                window['status_bar'].update("GTVelocityStreamer ok")

        else:
            logging.warning("Radio radio_velocity_source not valid")
       
        velocity_probe = np.round(velocity_probe, decimals=1)



        window['text_velocity'].update("V=%5.1fmm/s"%(velocity_probe))

        output_volume = values['slider_output_volume']


        if event == 'btn_set_filter':

            # if values['input_filter_lowcutoff'].isnumeric():
            if isFloat(values['input_filter_lowcutoff']):
                lowcut = np.float(values['input_filter_lowcutoff'])
            window['input_filter_lowcutoff'].update(np.int(lowcut))

            if isFloat(values['input_filter_highcutoff']):
                highcut = np.float(values['input_filter_highcutoff'])
            window['input_filter_highcutoff'].update(np.int(highcut))

        if event == 'btn_set_textures_gain':
            if isFloat(values['input_textures_gain']):
                textures_gain = np.float(values['input_textures_gain'])
            window['input_textures_gain'].update(textures_gain)
        
            
        # if event == key and values[key].isnumeric() and values[key][-1] not in ('0123456789.-'):
        #     window[key].update(values[key][:-1])
        #     lowcut = np.float(values[key])
        # key = 'input_filter_highcutoff'
        # if event == key and values[key.isnumeric()] and values[key][-1] not in ('0123456789.-'):
        #     window[key].update(values[key][:-1])
        #     highcut = np.float(values[key])
        # key = 'input_textures_gain'
        # if event == key and values[key].isnumeric() and values[key][-1] not in ('0123456789.-'):
        #     window[key].update(values[key][:-1])
        #     textures_gain = np.float(values[key])
        #     print("atex gain", values['input_textures_gain'])


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
            # x = threading.Thread(target=thread_PlayTexture_callback, args=([device_index]), daemon=True)
            threads_texture.append(x)
            
            x.start()
            window['play_status_bar'].update("Playing")
            
        if event in ('btn_stop'):
            logging.info("Main    : Stop")

            flag_play_texture = False
            window['play_status_bar'].update("Stopped")


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
        if selected_texture != values['listbox_texture_input'][0] or event=='btn_set_textures_gain':
            N_texture = 512*4

            selected_texture = values['listbox_texture_input'][0]

            texture_file = textures_dictionary[selected_texture]
            freqs, spectrum = np.loadtxt(texture_file, delimiter=',', unpack=True)
            fs_spatial = np.int(np.ptp(freqs)+1)
            # print("fs_spatial",fs_spatial)
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
            # ax.set_xlabel('k [1/mm]')
            # ax.set_ylabel('A [a.u.]')
            # # ax.set_xlim(xmin=0, xmax=40)
            # ax.set_xlim(xmin=0, xmax=100)
            # ax.set_ylim(0,1)

            ax.set_xlabel('k [1/mm]')
            ax.set_ylabel('A [a.u.]')

            ax.set_xlim(xmin=0)

            agg_texture.draw()


    window.close()
