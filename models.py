import torch
import torch.nn as nn

class OsuModel(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.bp_emb_dim = hyperparams['bp_emb_dim']
        self.bn_emb_dim = hyperparams['bn_emb_dim']
        self.diff_emb_dim = hyperparams['diff_emb_dim']
        self.np_hidden_size = hyperparams['np_hidden_size']
        self.np_num_layers = hyperparams['np_num_layers']
        self.ns_pre_proj_size = hyperparams['ns_pre_proj_size']
        self.ns_hidden_size = hyperparams['ns_hidden_size']
        self.ns_num_layers = hyperparams['ns_num_layers']
        self.action_emb_dim = hyperparams['action_emb_dim']
        self.num_tokens = 256

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 8, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.beat_phase_emb = nn.Embedding(49, self.bp_emb_dim)
        self.beat_num_emb = nn.Embedding(4, self.bn_emb_dim)
        self.difficulty_proj = nn.Sequential(
            nn.Linear(1, self.diff_emb_dim),
            nn.ReLU(),
            nn.Linear(self.diff_emb_dim, self.diff_emb_dim),
            nn.ReLU()
        )
        self.action_emb = nn.Embedding(self.num_tokens, self.action_emb_dim)

        self.np_gru = nn.GRU(input_size=320 + self.bp_emb_dim + self.bn_emb_dim + self.diff_emb_dim,
                             hidden_size=self.np_hidden_size,
                             num_layers=self.np_num_layers,
                             batch_first=True,
                             bidirectional=True)
        self.np_proj_1 = nn.Linear(self.np_hidden_size*2, 128)
        self.np_proj_2 = nn.Linear(128, 1)

        self.ns_pre_proj = nn.Linear(128, self.ns_pre_proj_size)
        self.ns_gru = nn.GRU(input_size=320 + self.ns_pre_proj_size + self.bp_emb_dim + self.bn_emb_dim + self.diff_emb_dim + self.action_emb_dim,
                             hidden_size=self.ns_hidden_size,
                             num_layers=self.ns_num_layers,
                             batch_first=True,
                             bidirectional=False)
        self.ns_proj_1 = nn.Linear(self.ns_hidden_size, self.ns_hidden_size)
        self.ns_proj_2 = nn.Linear(self.ns_hidden_size, self.num_tokens)

    def np_forward(self, specs, beat_phases, beat_nums, difficulties):
        conv_outs = self.stack(specs)
        conv_outs = conv_outs.permute(0, 2, 1, 3).flatten(2, 3)
        bp_emb = self.beat_phase_emb(beat_phases)
        bn_emb = self.beat_num_emb(beat_nums)
        diff_proj = self.difficulty_proj(difficulties)

        # ========== Note Placement ========== #
        np_in = torch.cat([conv_outs, bp_emb, bn_emb, diff_proj], dim=-1)
        np_out, _ = self.np_gru(np_in)
        np_proj_1_out = self.gelu(self.np_proj_1(np_out))
        np_pred = self.sigmoid(self.np_proj_2(np_proj_1_out)).squeeze()

        return conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred

    def forward(self, specs, beat_phases, beat_nums, difficulties, actions):
        conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred = self.np_forward(
            specs, beat_phases, beat_nums, difficulties)
        
        # ========== Note Selection ========== #
        ns_pre_proj = self.gelu(self.ns_pre_proj(np_proj_1_out))
        action_emb = self.action_emb(actions)
        ns_in = torch.cat(
            [conv_outs, ns_pre_proj, bp_emb, bn_emb, diff_proj, action_emb], dim=-1)
        ns_out, _ = self.ns_gru(ns_in)

        ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
        ns_logit = self.ns_proj_2(ns_proj_1_out)

        return np_pred, ns_logit

    def infer(self, specs, beat_phases, beat_nums, difficulties):
        conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, _ = self.np_forward(
            specs, beat_phases, beat_nums, difficulties)

        ns_pre_proj = self.gelu(self.ns_pre_proj(np_proj_1_out))
        out = torch.zeros([specs.shape[0], specs.shape[2]])
        action_emb = self.action_emb(torch.zeros([specs.shape[0], 1], dtype=torch.long))
        last_hidden = torch.zeros([2, specs.shape[0], 256])
        ns_in = torch.cat([conv_outs, ns_pre_proj, bp_emb, bn_emb, diff_proj], dim=-1)

        for i in range(specs.shape[2]):
            ns_in_temp = torch.cat([ns_in[:, i:i+1], action_emb], -1) # N x 1 x C
            ns_out, last_hidden = self.ns_gru(ns_in_temp, last_hidden)
            ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
            ns_logit = self.ns_proj_2(ns_proj_1_out)
            ns_pred = ns_logit.argmax(dim=-1)
            out[:, i] = ns_pred
            action_emb = self.action_emb(ns_pred)
            
        return out

class ControlModel(nn.Module):
    '''
    Contrary to the default AutoOsu model, the control model utilizes audio only during note placement(onset detection).
    This is for testing how the audio context affects the generated actions.
    '''

    def __init__(self, hyperparams):
        super().__init__()
        self.bp_emb_dim = hyperparams['bp_emb_dim']
        self.bn_emb_dim = hyperparams['bn_emb_dim']
        self.diff_emb_dim = hyperparams['diff_emb_dim']
        self.np_hidden_size = hyperparams['np_hidden_size']
        self.np_num_layers = hyperparams['np_num_layers']
        self.ns_pre_proj_size = hyperparams['ns_pre_proj_size']
        self.ns_hidden_size = hyperparams['ns_hidden_size']
        self.ns_num_layers = hyperparams['ns_num_layers']
        self.action_emb_dim = hyperparams['action_emb_dim']
        self.num_tokens = 256

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 8, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5, 3), stride=(1, 2), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.beat_phase_emb = nn.Embedding(49, self.bp_emb_dim)
        self.beat_num_emb = nn.Embedding(4, self.bn_emb_dim)
        self.difficulty_proj = nn.Sequential(
            nn.Linear(1, self.diff_emb_dim),
            nn.ReLU(),
            nn.Linear(self.diff_emb_dim, self.diff_emb_dim),
            nn.ReLU()
        )
        self.action_emb = nn.Embedding(self.num_tokens, self.action_emb_dim)

        self.np_gru = nn.GRU(input_size=320 + self.bp_emb_dim + self.bn_emb_dim + self.diff_emb_dim,
                             hidden_size=self.np_hidden_size,
                             num_layers=self.np_num_layers,
                             batch_first=True,
                             bidirectional=True)
        self.np_proj_1 = nn.Linear(self.np_hidden_size*2, 128)
        self.np_proj_2 = nn.Linear(128, 1)
        self.ns_pre_proj = nn.Sequential(
            nn.Linear(1, self.ns_pre_proj_size),
            nn.ReLU(),
            nn.Linear(self.ns_pre_proj_size, self.ns_pre_proj_size),
            nn.ReLU()
        )

        self.ns_gru = nn.GRU(input_size=self.ns_pre_proj_size + self.bp_emb_dim + self.bn_emb_dim + self.diff_emb_dim + self.action_emb_dim,
                             hidden_size=self.ns_hidden_size,
                             num_layers=self.ns_num_layers,
                             batch_first=True,
                             bidirectional=False)
        self.ns_proj_1 = nn.Linear(self.ns_hidden_size, self.ns_hidden_size)
        self.ns_proj_2 = nn.Linear(self.ns_hidden_size, self.num_tokens)

    def np_forward(self, specs, beat_phases, beat_nums, difficulties):
        conv_outs = self.stack(specs)
        conv_outs = conv_outs.permute(0, 2, 1, 3).flatten(2, 3)
        bp_emb = self.beat_phase_emb(beat_phases)
        bn_emb = self.beat_num_emb(beat_nums)
        diff_proj = self.difficulty_proj(difficulties)

        # ========== Note Placement ========== #
        np_in = torch.cat([conv_outs, bp_emb, bn_emb, diff_proj], dim=-1)
        np_out, _ = self.np_gru(np_in)
        np_proj_1_out = self.gelu(self.np_proj_1(np_out))
        np_pred = self.sigmoid(self.np_proj_2(np_proj_1_out))

        return conv_outs, bp_emb, bn_emb, diff_proj, np_proj_1_out, np_pred

    def forward(self, specs, beat_phases, beat_nums, difficulties, actions):
        _, bp_emb, bn_emb, diff_proj, _, np_pred = self.np_forward(
            specs, beat_phases, beat_nums, difficulties)
        
        # ========== Note Selection ========== #
        ns_pre_proj = self.ns_pre_proj(np_pred)
        action_emb = self.action_emb(actions)
        ns_in = torch.cat(
            [ns_pre_proj, bp_emb, bn_emb, diff_proj, action_emb], dim=-1)
        ns_out, _ = self.ns_gru(ns_in)

        ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
        ns_logit = self.ns_proj_2(ns_proj_1_out)

        return np_pred.squeeze(), ns_logit

    def infer(self, specs, beat_phases, beat_nums, difficulties):
        _, bp_emb, bn_emb, diff_proj, _, np_pred = self.np_forward(
            specs, beat_phases, beat_nums, difficulties)

        ns_pre_proj = self.gelu(self.ns_pre_proj(np_pred))
        out = torch.zeros([specs.shape[0], specs.shape[2]])
        action_emb = self.action_emb(torch.zeros([specs.shape[0], 1], dtype=torch.long))
        last_hidden = torch.zeros([2, specs.shape[0], 256])
        ns_in = torch.cat([ns_pre_proj, bp_emb, bn_emb, diff_proj], dim=-1)

        for i in range(specs.shape[2]):
            ns_in_temp = torch.cat([ns_in[:, i:i+1], action_emb], -1) # N x 1 x C
            ns_out, last_hidden = self.ns_gru(ns_in_temp, last_hidden)
            ns_proj_1_out = self.gelu(self.ns_proj_1(ns_out))
            ns_logit = self.ns_proj_2(ns_proj_1_out)
            ns_pred = ns_logit.argmax(dim=-1)
            out[:, i] = ns_pred
            action_emb = self.action_emb(ns_pred)
            
        return out