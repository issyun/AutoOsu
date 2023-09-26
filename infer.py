import torch
import torchaudio
from pathlib import Path
import shutil
import zipfile
from models import OsuModel, ControlModel
from convert import BeatmapConverter
from utils import index_to_combination
import hydra

DEV = 'cpu'
# if torch.cuda.is_available():
#     DEV = 'cuda'
# else:
#     print('Warning: Model running on CPU.')

@hydra.main(version_base=None, config_path='.', config_name='inference_config')
def main(config):
    converter = BeatmapConverter(None, None, None, None)
    if config.model == 'default':
        model = OsuModel(config.hyperparams)
    elif config.model == 'control':
        model = ControlModel(config.hyperparams)
    
    print(f'Loading checkpoint {config.checkpoint}...', end='')
    model.load_state_dict(torch.load(Path(config.checkpoint))['model_state_dict'])
    model.to(DEV)
    print('Done.')

    beat_length = 60000 / config.bpm
    audio_fn = Path(config.audio_fn)
    print(f'Converting {audio_fn.name}...', end='')
    y, sr = torchaudio.load(audio_fn)
    y = y.mean(dim=0)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    specs, beat_phase, beat_num = converter.convert_audio(y, config.offset, beat_length)
    print('Done.')

    diff = torch.FloatTensor([config.difficulty]).expand(beat_phase.shape[0]).unsqueeze(-1)
    specs = specs.unsqueeze(0).to(DEV)
    beat_phase = beat_phase.unsqueeze(0).to(DEV)
    beat_num = beat_num.unsqueeze(0).to(DEV)
    diff = diff.unsqueeze(0).to(DEV)

    print('Generating beatmap...', end='')
    model.eval()
    with torch.inference_mode():
        out = model.infer(specs, beat_phase, beat_num, diff).squeeze()
    print('Done.')

    print('Writing beatmap...', end='')
    beatmap = []
    for i, token in enumerate(out):
        if token.item() > 0:
            key0, key1, key2, key3 = index_to_combination(token.item(), 4)
            beatmap.append([i * 10, key0, key1, key2, key3])

    beatmap_str_list = []
    long = [False, False, False, False]
    long_start = [[], [], [], []]
    long_end = [[], [], [], []]
    line_num = 0

    for action in beatmap:
        time = action[0]
        keys = action[1:]
        for key, token in enumerate(keys):
            if token > 0:
                xpos = 64 + key * 128
                if token == 2:
                    long_start[key].append(line_num)
                    beatmap_str_list.append(f'{xpos},192,{time},128,2,')
                    long[key] = True
                    line_num += 1
                elif token == 3 and long[key] == True:
                    long_end[key].append(time)
                    long[key] = False
                else:
                    beatmap_str_list.append(f'{xpos},192,{time},1,0,0:0:0:0:')
                    line_num += 1

    for start_key, l in enumerate(long_start):
        for i, line_num in enumerate(l):
            if len(long_end[start_key]) <= i:
                print('\nWarning: Long note start / end mismatch. Converting to normal note.')
                split = beatmap_str_list[line_num].split(',')
                beatmap_str_list[line_num] = f'{split[0]},{split[1]},{split[2]},1,0,0:0:0:0:'
                continue
            endtime = long_end[start_key][i]
            beatmap_str_list[line_num] = f'{beatmap_str_list[line_num]}{endtime}:0:0:0:0:'

    # Copy resources to temporary workspace
    temp_workspace = Path('./')
    copied_audio_fn = temp_workspace / audio_fn.name
    shutil.copy(audio_fn, copied_audio_fn)
    bg_fn = Path('resources/autoosu-background.png')
    copied_bg_fn = temp_workspace / 'background.png'
    shutil.copy(bg_fn, copied_bg_fn)

    # Write osu file
    osu_file = temp_workspace / f'AutoOsu - {config.output_title} [{config.output_version}].osu'
    with open(osu_file, 'w') as f:
        f.write('osu file format v14\n')
        f.write('\n')
        f.write('[General]\n')
        f.write(f'AudioFilename: {copied_audio_fn.name}\n')
        f.write('AudioLeadIn: 0\n')
        f.write('PreviewTime: -1\n')
        f.write('Countdown: 0\n')
        f.write('SampleSet: Normal\n')
        f.write('StackLeniency: 0.7\n')
        f.write('Mode: 3\n')
        f.write('LetterboxInBreaks: 0\n')
        f.write('SpecialStyle: 0\n')
        f.write('WidescreenStoryboard: 1\n')
        f.write('\n')
        f.write('[Editor]\n')
        f.write('DistanceSpacing: 0.8\n')
        f.write('BeatDivisor: 1\n')
        f.write('GridSize: 32\n')
        f.write('TimelineZoom: 1\n')
        f.write('\n')
        f.write('[Metadata]\n')
        f.write(f'Title:{config.output_title}\n')
        f.write(f'TitleUnicode:{config.output_title}\n')
        f.write('Artist:AutoOsu\n')
        f.write('ArtistUnicode:AutoOsu\n')
        f.write('Creator:AutoOsu\n')
        f.write(f'Version:{config.output_version}\n')
        f.write('Source:\n')
        f.write('Tags:\n')
        f.write('BeatmapID:0\n')
        f.write('BeatmapSetID:-1\n')
        f.write('\n')
        f.write('[Difficulty]\n')
        f.write('HPDrainRate:5\n')
        f.write('CircleSize:4\n')
        f.write('OverallDifficulty:5\n')
        f.write('ApproachRate:5\n')
        f.write('SliderMultiplier:1.4\n')
        f.write('SliderTickRate:1\n')
        f.write('\n')
        f.write('[Events]\n')
        f.write('//Background and Video events\n')
        f.write('0,0,"background.png",0,0\n')
        f.write('//Break Periods\n')
        f.write('//Storyboard Layer 0 (Background)\n')
        f.write('//Storyboard Layer 1 (Fail)\n')
        f.write('//Storyboard Layer 2 (Pass)\n')
        f.write('//Storyboard Layer 3 (Foreground)\n')
        f.write('//Storyboard Layer 4 (Overlay)\n')
        f.write('//Storyboard Sound Samples\n')
        f.write('\n')
        f.write('[TimingPoints]\n')
        f.write(f'{config.offset},{beat_length},4,2,1,40,1,0\n')
        f.write('\n')
        f.write('[HitObjects]\n')
        f.write('\n'.join(beatmap_str_list))

    # Zip files
    zip_fn = Path(f'AutoOsu - {config.output_title}.zip')
    with zipfile.ZipFile(zip_fn, 'w') as zipf:
        zipf.write(osu_file)
        zipf.write(copied_audio_fn)
        zipf.write(copied_bg_fn)

    zip_fn.rename(zip_fn.with_suffix('.osz'))
    osu_file.unlink()
    copied_audio_fn.unlink()
    copied_bg_fn.unlink()
    print('Done.')
    print(f'Generated {len(beatmap_str_list)} hit objects.')

if __name__ == '__main__':
    main()