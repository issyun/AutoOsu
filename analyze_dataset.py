from pathlib import Path
import time
import argparse
import torch
import pandas as pd
from utils import find_file_by_stem

def analyze_dataset(beatmap_path:Path, audio_path:Path, output_path:Path=Path('.')):
    beatmap_fns = sorted(list(beatmap_path.rglob('*.pt')))
    audio_fns = sorted(list(audio_path.glob('*.pt')))

    beatmap_stats = {'Filename': [], 
                     'Audio available': [], 
                     'Detected keys': [],
                     'Actions': [], 
                     'Onsets': [], 
                     'Beatmap data': [], 
                     'Difficulty': []}
    
    audio_stats = {'Filename': [],
                   'Beatmap available': [],
                   'Spectrogram': [],
                   'Beat phase': [],
                   'Beat number': []}
    
    print("Reading files...")
    for beatmap_fn in beatmap_fns:
        actions, onsets, beatmap, difficulty = torch.load(beatmap_fn).values()
        beatmap_stats['Filename'].append(beatmap_fn.name)
        beatmap_stats['Audio available'].append(find_file_by_stem(audio_fns, beatmap_fn.stem.split('-')[0]) != -1)
        beatmap_stats['Detected keys'].append(beatmap_fn.parent.stem)
        beatmap_stats['Actions'].append(actions.shape)
        beatmap_stats['Onsets'].append(onsets.shape)
        beatmap_stats['Beatmap data'].append(beatmap.shape)
        beatmap_stats['Difficulty'].append(difficulty)

    for audio_fn in audio_fns:
        specs, beat_phase, beat_num = torch.load(audio_fn).values()
        audio_stats['Filename'].append(audio_fn.name)
        audio_stats['Beatmap available'].append(find_file_by_stem(beatmap_fns, audio_fn.stem, substring=True) != -1)
        audio_stats['Spectrogram'].append(specs.shape)
        audio_stats['Beat phase'].append(beat_phase.shape)
        audio_stats['Beat number'].append(beat_num.shape)

    print("Writing CSV...")
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    bm_df = pd.DataFrame(beatmap_stats)
    bm_df.to_csv(output_path / f'beatmap-data-stats-{curr_time}.csv')
    a_df = pd.DataFrame(audio_stats)
    a_df.to_csv(output_path / f'audio-data-stats-{curr_time}.csv')
    print("Finished.")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--beatmap_path', '-b', type=str)
    argparser.add_argument('--audio_path', '-a', type=str)
    argparser.add_argument('--csv-out', '-o', type=str)
    args = argparser.parse_args()
    b_in = Path(args.beatmap_path)
    a_in = Path(args.audio_path)
    o = Path(args.csv_out) if args.csv_out else Path('.')

    analyze_dataset(b_in, a_in, o) 