from pathlib import Path
from datetime import datetime
import torch
from torchmetrics.classification import BinaryF1Score
from torcheval.metrics.functional import perplexity
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
                 valid_loader, device, experiment_config, hyperparams):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.checkpoint_path = Path(experiment_config.checkpoint_path)
        self.checkpoint_path.mkdir(exist_ok=True)
        self.np_fl_gamma = hyperparams.np_fl_gamma
        self.np_fl_weight = hyperparams.np_fl_weight
        self.ns_fl_gamma = hyperparams.ns_fl_gamma
        self.ns_fl_weight = hyperparams.ns_fl_weight
        self.np_loss_multiplier = 7
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

    def run(self, experiment_config, hyperparams):
        if experiment_config.log_to_wandb:
            wandb.init(project='AutoOsu',
                       name=experiment_config.run_name, config=hyperparams)

        self.model.to(self.device)
        for epoch in tqdm(range(self.start_epoch, self.start_epoch + hyperparams.num_epochs)):
            if experiment_config.train:
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

                    ns_acc = (ns_pred.argmax(dim=-1) ==
                              ns_label).float().mean()
                    if experiment_config.log_to_wandb:
                        wandb.log({'train_np_loss': np_loss.item(),
                                   'train_ns_loss': ns_loss.item(),
                                   'train_loss': batch_loss.item(),
                                   'train_acc': ns_acc.item(),
                                   'train_np_f1': self.f1(np_pred, np_label.int()).item()})

            if experiment_config.validate:
                self.model.eval()
                with torch.inference_mode():
                    valid_np_loss_sum = 0
                    valid_ns_loss_sum = 0
                    valid_loss_sum = 0
                    valid_np_f1_sum = 0
                    valid_ns_acc_sum = 0
                    valid_ns_onset_acc_sum = 0
                    valid_ns_ppl_sum = 0
                    for batch in tqdm(self.valid_loader, leave=False):
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
                            np_label, np_pred, self.np_fl_gamma, self.np_fl_weight) * self.np_loss_multiplier
                        ns_loss = multi_focal_loss(
                            ns_label, ns_pred, self.ns_fl_gamma, self.ns_fl_weight)

                        batch_loss = np_loss + ns_loss
                        valid_np_loss_sum += np_loss.item()
                        valid_ns_loss_sum += ns_loss.item()
                        valid_loss_sum += batch_loss.item()
                        ns_acc = (ns_pred.argmax(dim=-1) ==
                                  ns_label).float().mean()
                        is_onset = onsets.flatten() == 1
                        valid_ns_onset_acc_sum += (ns_pred[is_onset].argmax(
                            dim=-1) == ns_label[is_onset]).float().mean()
                        valid_ns_acc_sum += ns_acc.item()
                        valid_np_f1_sum += self.f1(np_pred,
                                                   np_label.int()).item()
                        valid_ns_ppl_sum += perplexity(ns_logit, actions_gt, ignore_index=0).item()

                    if experiment_config.log_to_wandb:
                        wandb.log({'valid_np_loss': valid_np_loss_sum / len(self.valid_loader),
                                   'valid_ns_loss': valid_ns_loss_sum / len(self.valid_loader),
                                   'valid_loss': valid_loss_sum / len(self.valid_loader),
                                   'valid_acc': valid_ns_acc_sum / len(self.valid_loader),
                                   'valid_ns_onset_acc': valid_ns_onset_acc_sum / len(self.valid_loader),
                                   'valid_np_f1': valid_np_f1_sum / len(self.valid_loader),
                                   'valid_ns_ppl': valid_ns_ppl_sum / len(self.valid_loader)})
                    
                    if experiment_config.print_metrics:
                        print(f'Model: {experiment_config.model}')
                        print(f'valid_np_loss: {valid_np_loss_sum / len(self.valid_loader)}')
                        print(f'valid_ns_loss: {valid_ns_loss_sum / len(self.valid_loader)}')
                        print(f'valid_loss: {valid_loss_sum / len(self.valid_loader)}')
                        print(f'valid_acc: {valid_ns_acc_sum / len(self.valid_loader)}')
                        print(f'valid_ns_onset_acc: {valid_ns_onset_acc_sum / len(self.valid_loader)}')
                        print(f'valid_np_f1: {valid_np_f1_sum / len(self.valid_loader)}')
                        print(f'valid_ns_ppl: {valid_ns_ppl_sum / len(self.valid_loader)}')

            if experiment_config.save_checkpoint:
                time = datetime.now().strftime('%m-%d-%H-%M-%S')
                checkpoint_path = Path(
                    self.checkpoint_path / f'{experiment_config.model}-{time}-epoch{epoch}.pt')
                self.save_checkpoint(epoch, checkpoint_path)

        if experiment_config.log_to_wandb:
            wandb.finish()


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig):
    experiment_config = config.experiment_config
    hyperparams = config.hyperparams

    base_set = OsuDataset(Path(experiment_config.beatmap_dir),
                          Path(experiment_config.audio_dir))
    generator = torch.Generator().manual_seed(experiment_config.random_seed)
    train_set, valid_set = torch.utils.data.random_split(
        base_set, [0.95, 0.05], generator)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=hyperparams.batch_size, shuffle=True, generator=generator, collate_fn=collate, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=12, shuffle=False, collate_fn=collate, drop_last=False)

    if experiment_config.model == 'default':
        model = OsuModel(hyperparams)
    elif experiment_config.model == 'control':
        model = ControlModel(hyperparams)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams.learning_rate)
    trainer = Trainer(model, optimizer, train_loader, valid_loader,
                      DEV, experiment_config, hyperparams)

    if experiment_config.load_checkpoint is not None:
        trainer.load_checkpoint(Path(experiment_config.load_checkpoint))
    trainer.run(experiment_config, hyperparams)


if __name__ == '__main__':
    main()
