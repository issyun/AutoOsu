from pathlib import Path
from datetime import datetime
import torch
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from dataset import OsuDataset, collate
from models import OsuModel, ControlModel
from loss import binary_focal_loss, multi_focal_loss

DEV = 'cpu'
if torch.cuda.is_available():
    DEV = 'cuda'
else:
    print('Warning: Model running on CPU.')

class Trainer():
    def __init__(self, model, optimizer, train_loader,
                 valid_loader, device, checkpoint_path: Path,
                 np_fl_gamma=2, np_fl_weight=0.8,
                 ns_fl_gamma=2, ns_fl_weight=0.8,
                 np_loss_multiplier=7):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.checkpoint_path = checkpoint_path
        checkpoint_path.mkdir(exist_ok=True)
        self.np_fl_gamma = np_fl_gamma
        self.np_fl_weight = np_fl_weight
        self.ns_fl_gamma = ns_fl_gamma
        self.ns_fl_weight = ns_fl_weight
        self.np_loss_multiplier = np_loss_multiplier
        self.start_epoch = 0
        self.f1 = BinaryF1Score().to(self.device)

    def load_checkpoint(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if (torch.is_tensor(v)):
                    state[k] = v.to(self.device)

    def save_checkpoint(self, epoch, fn):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, fn)

    def train(self, hyperparams=None, log_to_wandb=True, run_name=None):
        if log_to_wandb:
            wandb.init(project='AutoOsu', name=run_name, config=hyperparams)

        self.model.to(self.device)
        for epoch in tqdm(range(self.start_epoch, hyperparams['num_epochs'])):
            self.model.train()
            for batch in tqdm(self.train_loader, leave=False):
                specs, beat_phases, beat_nums, difficulties, onsets, actions = batch
                specs = specs.to(self.device)
                beat_phases = beat_phases.to(self.device)
                beat_nums = beat_nums.to(self.device)
                difficulties = difficulties.to(self.device)
                onsets = onsets.to(self.device)
                actions_gt = actions[:, 1:].to(self.device)
                actions_shifted = actions[:, :-1].to(self.device)

                np_pred, ns_logit = self.model(
                    specs, beat_phases, beat_nums, difficulties, actions_shifted)

                np_pred = torch.reshape(np_pred, [-1])
                np_label = torch.reshape(onsets, [-1])

                ns_pred = torch.reshape(
                    ns_logit, [-1, ns_logit.shape[-1]]).softmax(dim=-1)
                ns_label = torch.reshape(actions_gt, [-1])

                np_loss = binary_focal_loss(
                    np_label, np_pred, self.np_fl_gamma, self.np_fl_weight)
                ns_loss = multi_focal_loss(
                    ns_label, ns_pred, self.ns_fl_gamma, self.ns_fl_weight)

                batch_loss = np_loss * self.np_loss_multiplier + ns_loss
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                ns_acc = (ns_pred.argmax(dim=-1) == ns_label).float().mean()
                if log_to_wandb:
                    wandb.log({'train_np_loss': np_loss.item(),
                               'train_ns_loss': ns_loss.item(),
                               'train_loss': batch_loss.item(),
                               'train_acc': ns_acc.item(),
                               'train_np_f1': self.f1(np_pred, np_label.int()).item()})

            self.model.eval()
            with torch.inference_mode():
                valid_np_loss_sum = 0
                valid_ns_loss_sum = 0
                valid_loss_sum = 0
                valid_np_f1_sum = 0
                valid_ns_acc_sum = 0
                for batch in tqdm(self.valid_loader, leave=False):
                    specs, beat_phases, beat_nums, difficulties, onsets, actions = batch
                    specs = specs.to(self.device)
                    beat_phases = beat_phases.to(self.device)
                    beat_nums = beat_nums.to(self.device)
                    difficulties = difficulties.to(self.device)
                    onsets = onsets.to(self.device)
                    actions_gt = actions[:, 1:].to(self.device)
                    actions_shifted = actions[:, :-1].to(self.device)

                    np_pred, ns_logit, _ = self.model(
                        specs, beat_phases, beat_nums, difficulties, actions_shifted)

                    np_pred = torch.reshape(np_pred, [-1])
                    np_label = torch.reshape(onsets, [-1])

                    ns_pred = torch.reshape(
                        ns_logit, [-1, ns_logit.shape[-1]]).softmax(dim=-1)
                    ns_label = torch.reshape(actions_gt, [-1])

                    np_loss = binary_focal_loss(
                        np_label, np_pred, self.np_fl_gamma, self.np_fl_weight) * self.np_loss_multiplier
                    ns_loss = multi_focal_loss(
                        ns_label, ns_pred, self.ns_fl_gamma, self.ns_fl_weight)

                    batch_loss = np_loss + ns_loss
                    valid_np_loss_sum += np_loss.item()
                    valid_ns_loss_sum += ns_loss.item()
                    valid_loss_sum += batch_loss.item()
                    ns_acc = (ns_pred.argmax(dim=-1) ==
                              ns_label).float().mean()
                    valid_ns_acc_sum += ns_acc.item()
                    valid_np_f1_sum += self.f1(np_pred, np_label.int()).item()

                if log_to_wandb:
                    wandb.log({'valid_np_loss': valid_np_loss_sum / len(self.valid_loader),
                               'valid_ns_loss': valid_ns_loss_sum / len(self.valid_loader),
                               'valid_loss': valid_loss_sum / len(self.valid_loader),
                               'valid_acc': valid_ns_acc_sum / len(self.valid_loader),
                               'valid_np_f1': valid_np_f1_sum / len(self.valid_loader)})

            time = datetime.now().strftime('%m-%d-%H-%M-%S')
            checkpoint_path = Path(
                self.checkpoint_path / f'{time}-epoch{epoch}.pt')
            self.save_checkpoint(epoch, checkpoint_path)

        if log_to_wandb:
            wandb.finish()

@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig):
    hyperparams = OmegaConf.to_container(config.hyperparams)
    base_set = OsuDataset(Path(config.train.beatmap_dir), Path(config.train.audio_dir))
    generator = torch.Generator().manual_seed(config.train.random_seed)
    train_set, valid_set = torch.utils.data.random_split(
        base_set, [0.95, 0.05], generator)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=hyperparams['batch_size'], shuffle=True, generator=generator, collate_fn=collate, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=12, shuffle=False, collate_fn=collate, drop_last=False)

    if config.train.model == 'default':
        model = OsuModel(hyperparams)
    elif config.train.model == 'control':
        model = ControlModel(hyperparams)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams['learning_rate'])
    trainer = Trainer(model, optimizer, train_loader, valid_loader,
                    device=DEV, checkpoint_path=Path(config.train.checkpoint_path),
                    np_fl_gamma=hyperparams['np_fl_gamma'], np_fl_weight=hyperparams['np_fl_weight'],
                    ns_fl_gamma=hyperparams['ns_fl_gamma'], ns_fl_weight=hyperparams['ns_fl_weight'])
    trainer.train(log_to_wandb=config.train.log_to_wandb, 
                  run_name = config.train.run_name,
                  hyperparams=hyperparams)

if __name__ == '__main__':
    main()