import random
import torch

class OsuDataset:
    """
    The dataset requires beatmaps and audio to be converted beforehand and saved as .pt files.
    Once initialized, all of the data will be loaded onto memory.
    GETITEM: specs, beat_phase, beat_num, difficulty, onsets, actions
    """

    def __init__(self, beatmap_path, audio_path):
        self.beatmap_fns = sorted(list(beatmap_path.glob('*.pt')))
        self.audio_fns = sorted(list(audio_path.glob('*.pt')))
        print('Loading dataset...', end='')
        self.beatmap_stems = [fn.stem.split('-')[0] for fn in self.beatmap_fns]
        self.beatmaps = [torch.load(fn) for fn in self.beatmap_fns]
        self.audio_stems = [fn.stem for fn in self.audio_fns]
        self.audio = [torch.load(fn) for fn in self.audio_fns]
        print('Done')

    def __len__(self):
        return len(self.beatmaps)

    def __getitem__(self, idx):
        try:
            audio_idx = self.audio_stems.index(self.beatmap_stems[idx])
        except ValueError:
            raise FileNotFoundError(
                f'Audio file not found for {self.beatmap_stems[idx]}')

        actions, onsets, _, difficulty = self.beatmaps[idx].values()
        specs, beat_phase, beat_num = self.audio[audio_idx].values()

        # randomly slice data to 30s
        if specs.shape[1] > 3001:
            start = random.randint(0, specs.shape[1] - 3001)
            actions = actions[start:start+3001]
            onsets = onsets[start+1:start+3001]
            specs = specs[:, start+1:start+3001, :]
            beat_phase = beat_phase[start+1:start+3001]
            beat_num = beat_num[start+1:start+3001]
            difficulty = (torch.FloatTensor(
                [difficulty]) * 0.2).expand(3000).unsqueeze(-1)
        else:
            raise IndexError(
                f'Beatmap shorter than 30s: {self.beatmap_stems[idx]}')

        return specs, beat_phase, beat_num, difficulty, onsets, actions


def collate(batch):
    specs = []
    beat_phases = []
    beat_nums = []
    difficulties = []
    onsets = []
    actions = []

    for x in batch:
        specs.append(x[0])
        beat_phases.append(x[1])
        beat_nums.append(x[2])
        difficulties.append(x[3])
        onsets.append(x[4])
        actions.append(x[5])

    specs = torch.stack(specs)
    beat_phases = torch.stack(beat_phases)
    beat_nums = torch.stack(beat_nums)
    difficulties = torch.stack(difficulties)
    onsets = torch.stack(onsets)
    actions = torch.stack(actions)

    return specs, beat_phases, beat_nums, difficulties, onsets, actions
