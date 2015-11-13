import pyaudio
import wave
import sys
import Queue

record_on = False
playback_on = False
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

recorded_frames = Queue.Queue()

def callback_play_sine(in_data, frame_count, time_info, status):
    if record_on:
        global recorded_frames
        recorded_frames.put(in_data)

    if playback_on:
        left_channel_data = mysine.next_block(frame_count) * MAX_INT24 * gain
        right_channel_data = ((np.random.rand(frame_count) * 2) - 1) * MAX_INT24 * gain
        data = interleave_channels(max_nr_of_channels, (left_output_channel, left_channel_data), (right_output_channel, right_channel_data))
        data = convert_int32_to_24bit_bytestream(data)
    else:
        data = np.zeros(frame_count*max_nr_of_channels).tostring()

    if stop_callback:
        callback_flag = pyaudio.paComplete
    else:
        callback_flag = pyaudio.paContinue

    return data, callback_flag



# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

raw_input("Press to Record: ")
record_on=True
raw_input("Press to Stop")
record_on=False
print recorded_frames
raw_input("Press to Play")
playback_on=True
raw_input("Press to Stop")
playback_on=False