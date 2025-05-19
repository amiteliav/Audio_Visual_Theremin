import numpy as np
import sounddevice as sd
import threading
import scipy.signal


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_to_midi(note: str) -> int:
    name = note[:-1]
    octave = int(note[-1])
    index = NOTE_NAMES.index(name)
    return (octave + 1) * 12 + index

def midi_to_freq(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))

def generate_pitch_dict(start_note: str, end_note: str) -> dict:
    start_midi = note_to_midi(start_note)
    end_midi = note_to_midi(end_note)
    pitch_dict = {}
    for midi in range(start_midi, end_midi + 1):
        note_name = NOTE_NAMES[midi % 12] + str((midi // 12) - 1)
        pitch_dict[note_name] = midi_to_freq(midi)
    return pitch_dict


def get_pitch_from_xy(x, y, frame_width, pitch_area, f_min, f_max):
    """
    Maps x position (0 to frame_width) to frequency between f_min and f_max.
    Note that we should use logarithmic mapping for frequency.
    The detection area is defined by x_start and x_end.
    """
    # Clamp x to be within the frame width
    if x < 0: x = 0
    if x > frame_width: x = frame_width

    if not (pitch_area['x_start'] <= x <= pitch_area['x_end'] and
            pitch_area['y_start'] <= y <= pitch_area['y_end']):
        return None  # Outside pitch area

    min_log = np.log2(f_min)
    max_log = np.log2(f_max)

    ratio = (x - pitch_area['x_start']) / (pitch_area['x_end'] - pitch_area['x_start'])
    log_freq = min_log + ratio * (max_log - min_log)
    return 2 ** log_freq

def get_amplitude_from_xy(x, y, frame_height, volume_area, min_amp=0.0, max_amp=0.95):
    """
    Maps y position (0 at top of frame to frame_height at bottom) to amplitude (min_amp to max_amp).
    The volume detection area is defined by volume_area.
    Returns None if (x, y) is outside the volume area.
    """
    # Clamp y within the frame height
    y = max(0, min(frame_height, y))

    if not (volume_area['x_start'] <= x <= volume_area['x_end'] and
            volume_area['y_start'] <= y <= volume_area['y_end']):
        return None  # Outside volume area

    # Invert y-axis: top is high volume, bottom is low
    ratio = 1.0 - (y - volume_area['y_start']) / (volume_area['y_end'] - volume_area['y_start'])
    amplitude = min_amp + ratio * (max_amp - min_amp)

    return amplitude

class ToneGenerator:
    def __init__(self, pitch_range, samplerate=22050):
        self.fs = samplerate

        self.freq = 440.0  # the default frequency (A4), changes as needed
        self.freq_thr = 1  # threshold for frequency change, as percentage
        self.alpha = 0.3  # smoothing factor for frequency change
        self.phase = 0.0
        self.amplitude = 0.75  # Default amp, can be changed dynamically

        self.depth = None        # depth value from depth estimation, or None to disable
        self.num_harmonics = 1   # number of harmonics to generate
        self.max_harmonics = 15  # maximum number of harmonics
        self.harmonics_enabled = False  # A flag whether to enable harmonics

        self.lock = threading.Lock()  # thread-safe access to freq

        self.pitch_range = pitch_range
        self.pitch_dict = generate_pitch_dict(*self.pitch_range)
        self.f_min = min(self.pitch_dict.values())
        self.f_max = max(self.pitch_dict.values())

        self.blocksize = 1024
        self.half_block = self.blocksize // 2

        self.prev_wave = np.zeros(self.half_block, dtype=np.float32)
        self.window = np.hanning(self.blocksize)

        self.clarinet_amp = [1.0, 0.75, 0.5, 0.25, 0.14, 0.09, 0.06, 0.04, 0.03]  # amplitudes for 1st, 3rd, ..., 11th

        self.stream = sd.OutputStream(
            channels=2,
            samplerate=self.fs,
            blocksize=self.half_block,  # Or 1024, 2048, 4096 etc.
            callback=self.callback,
        )
        self.stream.start()


    def stop(self):
        self.stream.stop()
        self.stream.close()

    def update_pitch_from_xy(self, x, y, frame_width, pitch_area):
        """
        Maps the x and y coordinates to a frequency in the range of f_min and f_max.
        """
        freq = get_pitch_from_xy(x, y, frame_width, pitch_area, self.f_min, self.f_max)
        self.set_frequency(freq)
        return freq

    def update_amplitude_from_xy(self, x, y, frame_height, pitch_area):
        """
        Maps the x and y coordinates to an amplitude value
        """
        amp = get_amplitude_from_xy(x, y, frame_height, pitch_area)
        self.set_amplitude(amp)

    def set_frequency(self, new_freq):
        """
        Set the new frequency for the tone generator.
        Changes the frequency only if the change is significant (> freq_thr in percentage).
        Changes the frequency in a smooth way using alpha (small alpha->mostly old value).

        This frequency will be the f0 (fundamental frequency) of the sound.
        """
        with self.lock:
            if new_freq is None:
                return  # Outside pitch area -> no change to self.freq

            relative_change = abs(self.freq - new_freq) / self.freq * 100

            # Update frequency only if changes significantly
            if relative_change > self.freq_thr:
                # Smoothing toward target frequency
                self.freq = (1 - self.alpha) * self.freq + self.alpha * new_freq

    def set_amplitude(self, amplitude):
        """
        Change the amplitude of the sound.
        """
        if amplitude is None:
            return  # Outside pitch area -> no change

        # Exponential smoothing toward target frequency
        self.amplitude = (1 - self.alpha) * self.amplitude + self.alpha * amplitude

    def depth_to_num_harmonics(self, depth):
        """
        Maps the depth value to the number of harmonics.
        Sets self.num_harmonics and self.harmonics_enabled

        :param depth: A value from the depth map or None if no depth detected

        """
        if depth is None: # No depth detected
            self.harmonics_enabled = False
            self.num_harmonics = 1  # No harmonics (N=1 -> only the fundamental frequency)
        else:  # TODO: still need to think about it
            self.harmonics_enabled = True
            # Example: map depth in [0, 1] to [1, max_harmonics]
            N = int(1 + depth * (self.max_harmonics - 1))
            self.num_harmonics = self.max_harmonics

    def generate_sound(self, t):
        """
        This method generates the sound wave based on the current frequency and amplitude.
        It uses the sine function to create a pure tone, and can also generate harmonics
        It generates a full-block and callback() will do the overlap-add
        """
        f = self.freq

        if self.harmonics_enabled:
            # Create a sine wave with harmonics
            wave = np.zeros_like(t, dtype=np.float32)

            # for i in range(self.num_harmonics):
            #     n = 2 * i + 1  # Generate 1st, 3rd, 5th, ... (odd harmonics -> clarinet-like sound)
            #     wave += (1 / n**2) * np.sin(2 * np.pi * f * n * t)

            for i in range(self.num_harmonics):
                n = 2 * i + 1  # Generate 1st, 3rd, 5th, ... (odd harmonics -> clarinet-like sound)
                if i < len(self.clarinet_amp):
                    amp = self.clarinet_amp[i]
                else:  # if we have more harmonics than defined in clarinet_amp
                    amp = 1 / n ** 2  # fallback
                wave += amp * np.sin(2 * np.pi * f * n * t)
        else:
            # Create only a sine wave with the target frequency
            wave = np.sin(2 * np.pi * f * t)

        # Apply amplitude
        wave /= np.max(np.abs(wave))  # normalize
        wave = (self.amplitude * wave).astype(np.float32)

        return wave


    def callback(self, outdata, frames, time, status):
        """
        Audio callback function to generate sound.
        This function fills the outdata in-place with the audio samples

        Note: callback expects to receive (data, frames, time, status)
        """
        t = (np.arange(self.blocksize) + self.phase) / self.fs
        with self.lock:
            f = self.freq

        wave = self.generate_sound(t)

        # Apply a window
        wave *= self.window

        # Overlap-add: sum previous tail with first half of current
        out_block = self.prev_wave + wave[:self.half_block]

        # Save the second half for the next callback
        self.prev_wave = wave[self.half_block:]

        # Convert to stereo by repeating the wave for both channels
        outdata[:] = np.tile(out_block[:, np.newaxis], (1, 2))  # Repeat wave across both channels
        self.phase = (self.phase + self.half_block) % self.fs


if __name__ == "__main__":
    print("Hello World")
    print("-----------------------")