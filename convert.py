from sys import platform
from pathlib import Path
from time import localtime, strftime
from math import floor
import torch
import torch.nn as nn
import torchaudio
from tqdm.auto import tqdm
from utils import round_base, combination_to_index, move_file, find_file_by_stem

is_macos = platform == 'darwin'
torch.set_printoptions(sci_mode=False)


class BeatmapConverter:
    def __init__(self,
                 osu_path: Path,
                 audio_path: Path,
                 beatmap_output_path: Path,
                 audio_output_path: Path,
                 n_fft_list: list = [1024, 2048, 4096],
                 hop_ms: int = 10,
                 beat_division=48):

        self.osu_path = osu_path
        self.audio_path = audio_path
        self.beatmap_output_path = beatmap_output_path
        self.audio_output_path = audio_output_path
        self.hop_ms = hop_ms
        self.hop_length = int(44100 * (hop_ms / 1000))
        self.melspec_converters = [torchaudio.transforms.MelSpectrogram(sample_rate=44100,
                                                                        n_fft=n_fft,
                                                                        hop_length=self.hop_length,
                                                                        f_max=11000,
                                                                        n_mels=80,
                                                                        power=2)
                                   for n_fft in n_fft_list]
        self.beat_division_length = round(1 / beat_division, 5)

    def get_beat_phase(self, time, offset: float, beat_length: float):
        beat_phase = ((time - offset) % beat_length) / beat_length
        return round_base(beat_phase, self.beat_division_length, index=True)

    def get_beat_num(self, time, beat_length, meter, anchor):
        beat_num = ((time - anchor) / beat_length) % meter
        if isinstance(beat_num, torch.Tensor):
            beat_num = beat_num.floor()
        elif isinstance(beat_num, int) or isinstance(beat_num, float):
            beat_num = floor(beat_num)
        return beat_num

    def parse_beatmap(self, fn):
        """
        Extract beatmap beat objects from an .osu file.
        RETURNS: Beatmap: Tensor([num_notes X 3(time, key_number, note_type)]),
                 num_keys, offset, beat_length, difficulty
                 (-1 if error)
        """

        with open(fn, mode='r', encoding='utf-8') as f:
            raw_content = f.read().splitlines()

        # Get difficulty (round to .2)
        difficulty = fn.name.split('-')[1]
        difficulty = round_base(
            float(difficulty[:1] + '.' + difficulty[1:]), 0.2)

        # Get timing points
        timing_points = []
        # Read everything until next section
        i = raw_content.index('[TimingPoints]') + 1
        while raw_content[i] != '' and raw_content[i][0] != '[':
            timing_points.append(raw_content[i])
            i += 1

        # Check if multiple BPMs exist
        beat_lengths = {
            float(tp.split(',')[1]) for tp in timing_points if float(tp.split(',')[1]) > 0}
        if len(beat_lengths) > 1:
            return -1, -1, -1, -1, -1

        # Make sure beatmap contains only one meter of 4
        meters = {int(tp.split(',')[2])
                  for tp in timing_points if float(tp.split(',')[1]) > 0}
        if len(meters) > 1:
            return -1, -1, -1, -1, -1
        if list(meters)[0] != 4:
            return -1, -1, -1, -1, -1

        # Get offset and beat length
        offset = float(timing_points[0].split(',')[0])
        beat_length = beat_lengths.pop()

        # Parse and convert beat objects
        beatmap_start_index = raw_content.index('[HitObjects]')
        beatmap = raw_content[beatmap_start_index + 1:]

        obj_list = []
        xpos_set = set()
        for obj in beatmap:
            obj_split = obj.split(',')
            time = int(obj_split[2])
            xpos = int(obj_split[0])
            xpos_set.add(xpos)

            if int(obj_split[3]) > 6:  # If note is long note...
                end_time = int(obj_split[5].split(':', 1)[0])
                obj_list.append([time, xpos, 2])
                obj_list.append([end_time, xpos, 3])
            else:
                obj_list.append([time, xpos, 1])

        # Convert X-position to key number
        xpos_list = sorted(xpos_set)
        num_keys = len(xpos_list)
        xpos2num = {xpos: num for num, xpos in enumerate(xpos_list)}
        obj_list = [[obj[0], xpos2num[obj[1]], obj[2]] for obj in obj_list]

        # Convert to tensor and sort by note time
        obj_tensor = torch.tensor(obj_list, dtype=torch.float32)
        obj_tensor = obj_tensor[obj_tensor[:, 0].argsort()]

        return obj_tensor, num_keys, offset, beat_length, difficulty

    def convert_audio(self, y, offset, beat_length, eps=1e-9):
        """
        Converts audio into 3-channel mel-spectrogram with context windows.
        INPUT: waveform of sr=44100, offset, beat length
        RETURNS: Spectrogram: Tensor([num_timesteps, 80, 3]),
                 Beat phase: Tensor([num_timesteps]),
                 Beat num: Tensor([num_timesteps])
        """

        # Multiple-timescale STFT
        specs = []
        for converter in self.melspec_converters:
            melspec = converter(y + eps)
            specs.append(torch.log(melspec.T))
        specs = torch.stack(specs, dim=-1)  # len X 80 X 3

        # Create beat phase tensor
        beat_phase = self.get_beat_phase(torch.arange(
            len(specs)) * self.hop_ms, offset, beat_length)
        beat_num = self.get_beat_num(torch.arange(
            len(specs)) * self.hop_ms, beat_length, 4, offset)

        return specs, beat_phase, beat_num

    def quantize_beatmap(self, beatmap, num_timesteps, num_keys):
        """
        Quantizes beat objects to the grid of hop_ms.
        INPUT: Beatmap: Tensor([num_notes X 3(time, key_number, note_type)])
        RETURNS: Actions: Tensor([num_timesteps, 1])
                 Onset: Tensor([num_timesteps, 1])
        """

        # Quantize timings to hop_ms
        beatmap_new = beatmap.clone()
        timesteps = round_base(beatmap_new[:, 0], self.hop_ms) / self.hop_ms
        beatmap_new[:, 0] = timesteps

        # Create action tensor whose length matches with spectrogram
        actions = torch.zeros([num_timesteps, num_keys])
        for obj in beatmap_new:
            timestep, key_number, note_type = obj.tolist()
            actions[int(timestep), int(key_number)] = note_type

        actions = torch.tensor([combination_to_index(
            obj.tolist(), num_keys) for obj in actions])
        onsets = actions.bool().int()

        return actions, onsets

    def convert(self):
        audio_suffixes = {'.mp3', '.MP3', '.wav', '.WAV', '.ogg', '.OGG'}
        audio_fns = sorted([p for p in self.audio_path.glob(
            '**/*') if p.suffix in audio_suffixes])
        osu_fns = sorted(list(self.osu_path.glob('*.osu')))

        # Create needed paths
        excluded_osu_path = self.osu_path / 'excluded_osu'
        excluded_osu_path.mkdir(exist_ok=True)
        num_keys_paths = []  # Separate converted beatmaps by num_keys

        log = open(self.beatmap_output_path /
                   (strftime('conversion-log-%Y-%m-%d-%H-%M-%S', localtime()) + '.txt'), 'w')

        for osu_fn in tqdm(osu_fns):
            # Parse beatmap notes
            beatmap, num_keys, offset, beat_length, difficulty = self.parse_beatmap(
                osu_fn)
            if (num_keys == -1):
                log.write(
                    f'ERROR {osu_fn.name}: Incompatible beatmap. Skipping conversion.\n')
                move_file(osu_fn, excluded_osu_path / osu_fn.name)
                continue

            # Check if num_keys path already exists
            num_keys_path = self.beatmap_output_path / f'{num_keys}keys/'
            if num_keys_path not in num_keys_paths:
                num_keys_path.mkdir(exist_ok=True)
                num_keys_paths.append(num_keys_path)

            # Check if corresponding audio has already been converted
            audio_stem = osu_fn.stem.split('-')[0]
            converted_audio_fn = find_file_by_stem(
                list(self.audio_output_path.glob('*.pt')), audio_stem)

            num_timesteps = 0
            if converted_audio_fn == -1:  # If audio hasn't been converted...
                # Load audio with OS-specific backend
                audio_fn = find_file_by_stem(audio_fns, audio_stem)
                if audio_fn == -1:
                    log.write(
                        f'ERROR {osu_fn.name}: Audio file not found. Skipping conversion.\n')
                    move_file(osu_fn, excluded_osu_path / osu_fn.name)
                    continue

                if is_macos:
                    y, sr = torchaudio.load(audio_fn, backend='ffmpeg')
                else:
                    y, sr = torchaudio.load(audio_fn)

                # Mono and resample
                y = y.mean(dim=0)
                if sr != 44100:
                    log.write(
                        f'WARN Sampling rate of file {audio_fn.name} is {sr}: Resampling to 44100.\n')
                    y = torchaudio.functional.resample(y, sr, 44100)

                # Convert audio and save
                specs, beat_phase, beat_num = self.convert_audio(
                    y, offset, beat_length)
                num_timesteps = len(specs)
                torch.save({'specs': specs, 'beat_phase': beat_phase, 'beat_num': beat_num},
                           (self.audio_output_path / f'{audio_fn.stem}.pt'))

            else:  # If audio has already been converted...
                specs, beat_phase, beat_num = torch.load(
                    converted_audio_fn).values()
                num_timesteps = len(specs)

            # Quantize beatmap and save
            actions, onsets = self.quantize_beatmap(
                beatmap, num_timesteps, num_keys)
            if not len(specs) == len(beat_phase) == len(beat_num) == len(actions) == len(onsets):
                log.write(
                    f'ERROR {osu_fn.name}: Features dimensions mismatch. Skipping conversion.\n')
                move_file(osu_fn, excluded_osu_path / osu_fn.name)
                continue

            torch.save({'actions': actions, 'onsets': onsets, 'beatmap': beatmap, 'difficulty': difficulty},
                       (num_keys_path / f'{osu_fn.stem}.pt'))

        log.close()
