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


from texture_vibration_generator import *




# the global buffer variable must be defined here.. so callback must be in this file
def callback(in_data, frame_count, time_info, status):

    global zf
    global b,a
    global buffer
    global next_frame
    global window
    

    if 1:
        next_frame = 0
        velocity_probe = cursor.speed()/500 +10
        fs_temporal = fs_spatial*velocity_probe
        
        text = "Velocity %0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal)
        print(text)
        window['text_velocity'].update(text)
        append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
        
    
    frame = np.copy(buffer.extractleft(frame_count))

#         sig = butter_lowpass_filter(sig, cutoff, fs_audio, order=5)
    frame, zf = signal.lfilter(b, a, frame, zi=zf)
    
    
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
        [sg.Canvas(key="canvas_texture_plot")],
        [sg.Checkbox('Measure Velocity', default=True), sg.Text("Velocity", key='text_velocity')],
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

    window["canvas_texture_plot"].set_tooltip('Texture spectrum')

    


    # Initializing spectrum texture
        # creating analytical spectrum texture
    if 1:
        N_texture = 512*1
        wavelength_texture = 1/10 # [mm]
        length_texture = 1

        spectrum_texture, fs_spatial = create_spectrum_texture(wavelength_texture, length_texture, N_texture, )
        
        fig = plt.figure()
        plt.plot(np.abs(spectrum_texture))


        draw_figure(window["canvas_texture_plot"].TKCanvas, fig)

    # Initializing Cursor
    if 1:
        cursor = Cursor()


    next_frame = 1
    # Initializing Audio
    if 1:
        # Streaming audio buffer to output.
        # With callback function

        # fs_audio = np.int(fs_temporal)
        # print(fs_temporal)
        # N = texture.size



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
        # while stream.is_active():
            
            
        #     if len(buffer) < N_audio_segment:
        #     if next_frame:
        #         next_frame = 0
        #         velocity_probe = cursor.speed()/500 +10
        #         fs_temporal = fs_spatial*velocity_probe
        #         print("\rVelocity %0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal), end="")
        #         append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
        # #         stream_continue = 0
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

        

    # Event Loop to process "events"
    while True:             
        event, values = window.read()

        # if event == sg.WIN_CLOSED or event == 'Exit':
        if event in ('Exit', None, 'Cancel'):
            break



        if next_frame:
            next_frame = 0
            velocity_probe = cursor.speed()/500 +10
            fs_temporal = fs_spatial*velocity_probe
            
            window['text_velocity'].update("Velocity %0.3f mm/s fs_temporal %0.3f"%(velocity_probe, fs_temporal))
            append_buffer_texture_signal(buffer, spectrum_texture, fs_spatial, velocity_probe, N_audio_segment, fs_audio, N_overlap )
            


    stream.stop_stream()
    stream.close()

    p.terminate

    window.close()
