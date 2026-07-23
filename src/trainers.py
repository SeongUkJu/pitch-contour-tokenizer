from typing import Tuple, List
from pathlib import Path

import wandb
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import random
from io import BytesIO

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class BaseTrainer:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim, 
                 scheduler: torch.optim.lr_scheduler,
                 trainset: Dataset, 
                 validset: Dataset, 
                 criterion: nn.Module, 
                 metric: List[nn.Module],
                 device: torch.device, 
                 save_dir: str, 
                 config: DictConfig,) -> None:
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)

        self.model = model
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.criterion = criterion
        self.metric = metric
        
        self.trainset = trainset
        self.validset = validset
        self.batch_size = config.train.get('batch_size', 64)

        self.global_step = 0
        self.num_iterations = config.train.get('num_iterations', 10000)
        self.eval_interval = config.train.get('eval_interval', 200)
        self.save_interval = config.train.get('save_interval', 2000)
        self.log_interval = config.train.get('log_interval', 10)

        self._micro = 0
        self.accum_steps = config.train.get('accum_steps', 1)
        self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(config.train.get('amp', 'off'), None)
        self.amp_enabled = (self.amp_dtype is not None) and torch.cuda.is_available()   # CPU면 자동 off(에뮬 회피)

        self.best_score, self.best_iteration = 100, 0
        self.recorder = {'Train': {}, 'Valid': {}, 'Test': {}}

        random_li = random.choices(range(len(validset)), k=4)
        self.samples = [(sample[0], sample[1]) for sample in [validset[sample_idx] for sample_idx in random_li]]

    def _eval_samples(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _calc_metric(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _train_batch(self):
        raise NotImplementedError("Subclasses must implement this method")

    def train(self):
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement this method")


class AETrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim, 
                 scheduler: torch.optim.lr_scheduler,
                 trainset: Dataset, 
                 validset: Dataset, 
                 criterion: nn.Module, 
                 metric: List[nn.Module],
                 device: torch.device, 
                 save_dir: str, 
                 config: DictConfig) -> None:
        super().__init__(model, optimizer, scheduler, trainset, validset, criterion, metric, device, save_dir, config)


    def get_plot_image(self, b_idx, target_contour, pred_contour, z):
        title = f'Step {self.global_step}'

        vq = False
        if 'VQ' in self.model.__class__.__name__:
            vq = True
            title += f', # of Codebooks {self.model.quantizer.n_e}, Codebook Dim {self.model.quantizer.e_dim}, $\\beta$={self.model.quantizer.vq_loss.beta}'
            z, zq = z

        median_seq = []
        median = self.model.subtract_median_from_x(target_contour[b_idx].unsqueeze(0))[1].squeeze().numpy()
        if target_contour[b_idx].shape[-1] // self.model.rf == 1:
            median_seq += [median]*target_contour[b_idx].shape[-1]
        else:
            for m in median:
                median_seq += [m]*self.model.rf

        fig = plt.figure(figsize=(20, 10))
        plt.plot(target_contour[b_idx,0], label='org')
        plt.plot(pred_contour[b_idx,0], label='rec')
        plt.plot(median_seq, label='med', alpha=0.5)

        if hasattr(self.validset, 'window_frame'):
            for i in range(self.validset.window_frame//self.model.rf):
                plt.axvline(i*self.model.rf, alpha=0.5)
            plt.axvline((i+1)*self.model.rf, alpha=0.5)
        elif hasattr(self.validset, 'window') and (self.validset.window//self.model.rf > 1):
            for i in range(self.validset.window//self.model.rf):
                plt.axvline(i*self.model.rf, alpha=0.5)
            plt.axvline((i+1)*self.model.rf, alpha=0.5)

        plt.title(title)
        plt.xlabel('Time frame(10ms)')
        plt.ylabel(f'Freq(normed with tonic)')
        plt.legend()
        plt.tight_layout()


        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_np = np.array(image)
        plt.close(fig)
        buf.close()

        return image_np


    def _eval_samples(self):
        if not self.config.wandb.get('project'):
            return
        samples = torch.stack([s[0] for s in self.samples]).to(self.device)
        weights = torch.stack([s[1] for s in self.samples]).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            if self.model.hparams.in_channels == 2:
                target, pred, _ = self.model(torch.cat([samples, weights], dim=-2))
                target_contour, pred_contour = target[0][:,0].unsqueeze(-2), pred[0][:,0].unsqueeze(-2)
            else:
                target, pred, _ = self.model(samples)
                target_contour, pred_contour= target, pred
            z = (target[-1].detach().cpu(),pred[-1].detach().cpu())

        if pred_contour.size() != target_contour.size():
            b,c,r,t = pred_contour.shape
            pr = pred_contour.reshape(b,c,r,t//self.model.rf,self.model.rf)
            tr = target_contour.unsqueeze(-2).reshape(b,c,1,t//self.model.rf,self.model.rf)
            squared_errors = torch.pow(pr-tr, 2) # b,c_in,rolled_size,t_comp,rf

            if self.criterion.use_weight:
                weight = weights.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//self.model.rf,self.model.rf)
                sqe = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
            else:
                sqe = torch.mean(squared_errors, dim=-1)

            min_ids = torch.argmin(sqe, dim=-2).unsqueeze(-2).unsqueeze(-1).expand(-1,-1,-1,-1,self.model.rf)
            pred_contour = pr.gather(dim=-3, index=min_ids).squeeze().reshape(b,c,-1)

        target_contour, pred_contour = target_contour.detach().cpu(), pred_contour.detach().cpu()

        img = []
        vis_log = {}
        for b_idx in range(samples.shape[0]):
            image_np = self.get_plot_image(b_idx, target_contour, pred_contour, z)
            img.append(image_np)
            vis_log[f"Vis/Sample{b_idx}"] = wandb.Image(Image.fromarray(image_np))
        wandb.log(vis_log, step=self.global_step)



    def _calc_metric(self,
                     target: torch.Tensor,
                     pred: torch.Tensor,
                     phase: str='Train',
                     **kwargs):
        if self.metric is None:
            return

        metric_inputs = {'pred': pred,
                        'target': target,
                        **kwargs}

        for m in self.metric:
            metric_name = m.__class__.__name__
            metric_value = m(**metric_inputs)
            self.recorder[f'{phase}'][f'{phase}/{metric_name}'] = metric_value.item()


    def _step_multi_channel_batch(self):
        raise NotImplementedError("Subclasses must implement this method")

    def _step_single_channel_batch(self,
                                    x: torch.Tensor,
                                    weight: torch.Tensor,
                                    meta: Tuple[torch.Tensor],
                                    phase: str='Train') -> Tuple[torch.Tensor]:
        target, pred, z = self.model(x) if not self.criterion.use_weight else self.model(x, mask=weight)
        recon_loss = self.criterion(pred, target, weight=weight)
        loss = recon_loss
        out = target, pred, loss
        
        self.recorder[f'{phase}'][f'{phase}/ReconLoss'] = recon_loss.item()

        return out


    def _train_batch(self, 
                     batch: Tuple[torch.Tensor]) -> float:
        self.recorder['Train'] = {}
        self.model.train()
        x, weight, meta = batch
        x, weight = x.to(self.device, non_blocking=True), weight.to(self.device, non_blocking=True)

        if self.model.hparams.in_channels != 1:
            target, pred, loss = self._step_multi_channel_batch(x, weight, meta, phase='Train')
        else:
            target, pred, loss = self._step_single_channel_batch(x, weight, meta, phase='Train')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.global_step % self.log_interval == 0:
            self._calc_metric(target, pred, phase='Train')
            self.recorder['Train'][f'Train/CurrentLR'] = self.optimizer.param_groups[0]['lr']
            self.recorder['Train'][f'Train/Loss'] = loss.item()
            wandb.log(self.recorder['Train'], step=self.global_step)

        return loss.item()


    def train(self) -> None:
        self.pbar = tqdm(total=self.num_iterations, desc='Iter')

        while self.global_step < self.num_iterations:
            self.trainset.resample_receptive_field()
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=4, pin_memory=True)

            for batch in train_loader:
                loss = self._train_batch(batch)

                self.global_step += 1
                self.pbar.update(1)
                self.pbar.set_description(f"Train Loss: {loss:.4f}")

                if self.global_step % self.eval_interval == 0:
                    self.evaluate(self.validset)

                if self.global_step % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.save_dir / f'{self.global_step}_iter.pt')

                if self.global_step >= self.num_iterations:
                    break

        print(f"Best Score: {self.best_score:.4f} at iteration {self.best_iteration}")

        self.pbar.close()


    def evaluate(self, 
                 dataset: Dataset, 
                 external_dataset: bool=False) -> None | float:
        phase = 'Test' if external_dataset else 'Valid'
        self.recorder[f'{phase}'] = {}
        self._eval_samples()

        self.model.eval()
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss, total_recon_loss = 0, 0
        total_metrics = [0 for _ in range(len(self.metric))] if self.metric else None

        with torch.inference_mode():
            for batch in data_loader:
                x, weight, meta = batch
                x, weight = x.to(self.device), weight.to(self.device)

                if self.model.hparams.in_channels != 1:
                    target, pred, loss = self._step_multi_channel_batch(x, weight, meta, phase=phase)
                else:
                    target, pred, loss = self._step_single_channel_batch(x, weight, meta, phase=phase)

                self._calc_metric(target, pred, phase=phase)

                total_loss += loss.item() * x.shape[0]
                total_recon_loss += self.recorder[f'{phase}'][f'{phase}/ReconLoss'] * x.shape[0]

                if total_metrics is not None:
                    for i, m in enumerate(self.metric):
                        total_metrics[i] += self.recorder[f'{phase}'][f'{phase}/{m.__class__.__name__}'] * x.shape[0]

        self.recorder[f'{phase}'][f'{phase}/Loss'] = total_loss / len(dataset)
        self.recorder[f'{phase}'][f'{phase}/ReconLoss'] = total_recon_loss / len(dataset)

        if total_metrics is not None:
            for i, m in enumerate(self.metric):
                self.recorder[f'{phase}'][f'{phase}/{m.__class__.__name__}'] = total_metrics[i] / len(dataset)


        if not external_dataset:
            wandb.log(self.recorder[f'{phase}'], step=self.global_step)
            self.pbar.set_description(f"Iter {self.global_step}/{self.num_iterations} | Val Loss: {self.recorder[f'{phase}'][f'{phase}/ReconLoss']:.4f}")
            
            # if valid_loss < self.best_score:
            if self.recorder[f'{phase}'][f'{phase}/ReconLoss'] < self.best_score:
                self.best_score = self.recorder[f'{phase}'][f'{phase}/ReconLoss'] # valid_loss
                self.best_iteration = self.global_step
                torch.save(self.model.state_dict(), self.save_dir/f'best_model.pt')
                print(f"Iter {self.global_step}, Best Score {self.best_score:.4f}")
                    
        else:
            print(f"Test Loss: {self.recorder[f'{phase}'][f'{phase}/ReconLoss']:.4f}")
            wandb.log(self.recorder[f'{phase}'], step=self.global_step)
            wandb.summary["Test/Loss"] = self.recorder[f'{phase}'][f'{phase}/ReconLoss']




class VQTrainer(BaseTrainer):
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim, 
                 scheduler: torch.optim.lr_scheduler,
                 trainset: Dataset, 
                 validset: Dataset, 
                 criterion: nn.Module, 
                 metric: List[nn.Module],
                 device: torch.device, 
                 save_dir: str, 
                 config: DictConfig) -> None:
        super().__init__(model, optimizer, scheduler, trainset, validset, criterion, metric, device, save_dir, config)


    def get_plot_image(self, b_idx, target_contour, pred_contour, z):
        title = f'Step {self.global_step}'

        vq = False
        if 'VQ' in self.model.__class__.__name__:
            vq = True
            title += f', # of Codebooks {self.model.quantizer.n_e}, Codebook Dim {self.model.quantizer.e_dim}, $\\beta$={self.model.quantizer.vq_loss.beta}'
            z, zq = z

        median_seq = []
        median = self.model.subtract_median_from_x(target_contour[b_idx].unsqueeze(0))[1].squeeze().numpy()
        if target_contour[b_idx].shape[-1] // self.model.rf == 1:
            median_seq += [median]*target_contour[b_idx].shape[-1]
        else:
            for m in median:
                median_seq += [m]*self.model.rf

        fig = plt.figure(figsize=(20, 10))
        plt.plot(target_contour[b_idx,0], label='org')
        plt.plot(pred_contour[b_idx,0], label='rec')
        plt.plot(median_seq, label='med', alpha=0.5)

        if hasattr(self.validset, 'window_frame'):
            for i in range(self.validset.window_frame//self.model.rf):
                plt.axvline(i*self.model.rf, alpha=0.5)
            plt.axvline((i+1)*self.model.rf, alpha=0.5)
        elif hasattr(self.validset, 'window') and (self.validset.window//self.model.rf > 1):
            for i in range(self.validset.window//self.model.rf):
                plt.axvline(i*self.model.rf, alpha=0.5)
            plt.axvline((i+1)*self.model.rf, alpha=0.5)

        plt.title(title)
        plt.xlabel('Time frame(10ms)')
        plt.ylabel(f'Freq(normed with tonic)')
        plt.legend()
        plt.tight_layout()


        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_np = np.array(image)
        plt.close(fig)
        buf.close()

        return image_np


    def _eval_samples(self):
        if not self.config.wandb.get('project'):
            return
        samples = torch.stack([s[0] for s in self.samples]).to(self.device)
        weights = torch.stack([s[1] for s in self.samples]).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            if self.model.hparams.in_channels == 2:
                target, pred, _, _ = self.model(torch.cat([samples, weights], dim=-2))
                target_contour, pred_contour = target[0][:,0].unsqueeze(-2), pred[0][:,0].unsqueeze(-2)
            else:
                target, pred, _, _ = self.model(samples)
                target_contour, pred_contour= target[0], pred[0]
            z = (target[-1].detach().cpu(),pred[-1].detach().cpu())

        if pred_contour.size() != target_contour.size():
            b,c,r,t = pred_contour.shape
            pr = pred_contour.reshape(b,c,r,t//self.model.rf,self.model.rf)
            tr = target_contour.unsqueeze(-2).reshape(b,c,1,t//self.model.rf,self.model.rf)
            squared_errors = torch.pow(pr-tr, 2) # b,c_in,rolled_size,t_comp,rf

            if self.criterion.use_weight:
                weight = weights.unsqueeze(-2).unsqueeze(-2).reshape(b,c,1,t//self.model.rf,self.model.rf)
                sqe = torch.sum(squared_errors * weight, dim=-1) / torch.sum(weight, dim=-1)
            else:
                sqe = torch.mean(squared_errors, dim=-1)

            min_ids = torch.argmin(sqe, dim=-2).unsqueeze(-2).unsqueeze(-1).expand(-1,-1,-1,-1,self.model.rf)
            pred_contour = pr.gather(dim=-3, index=min_ids).squeeze().reshape(b,c,-1)

        target_contour, pred_contour = target_contour.detach().cpu(), pred_contour.detach().cpu()

        img = []
        vis_log = {}
        for b_idx in range(samples.shape[0]):
            image_np = self.get_plot_image(b_idx, target_contour, pred_contour, z)
            img.append(image_np)
            vis_log[f"Vis/Sample{b_idx}"] = wandb.Image(Image.fromarray(image_np))
        wandb.log(vis_log, step=self.global_step)


    def _calc_metric(self,
                     target: torch.Tensor,
                     pred: torch.Tensor,
                     min_e: torch.Tensor, 
                     phase: str='Train',
                     **kwargs):
        metric_inputs = {'pred': pred,
                        'target': target,
                        'min_e': min_e,
                        **kwargs}

        for m in self.metric:
            metric_name = m.__class__.__name__
            metric_value = m(**metric_inputs)
            self.recorder[f'{phase}'][f'{phase}/{metric_name}'] = metric_value.item()


    def _step_multi_channel_batch(self,
                                    x: torch.Tensor,
                                    weight: torch.Tensor,
                                    meta: Tuple[torch.Tensor],
                                    phase: str='Train') -> Tuple[torch.Tensor]:
        raise NotImplementedError("No dataset type implemented")


    def _step_single_channel_batch(self,
                                    x: torch.Tensor,
                                    weight: torch.Tensor,
                                    meta: Tuple[torch.Tensor],
                                    phase: str='Train') -> Tuple[torch.Tensor]:
            target, pred, (vq_loss, _, _, ortho_loss), min_e = self.model(x) if not self.criterion.use_weight else self.model(x, mask=weight)
            recon_loss = self.criterion(pred, target, weight=weight)
            loss = recon_loss + vq_loss
            if ortho_loss is not None:
                loss += self.model.quantizer.ortho_reg * ortho_loss

            out = target, pred, min_e, loss
            
            self.recorder[f'{phase}'][f'{phase}/ReconLoss'] = recon_loss.item()
            self.recorder[f'{phase}'][f'{phase}/VQLoss'] = vq_loss.item()
            if ortho_loss is not None:
                self.recorder[f'{phase}'][f'{phase}/OrthoLoss'] = ortho_loss.item()

            return out


    def _train_batch(self, 
                     batch: Tuple[torch.Tensor]) -> float:
        self.recorder['Train'] = {}
        self.model.train()
        x, weight, meta = batch
        x, weight = x.to(self.device, non_blocking=True), weight.to(self.device, non_blocking=True)

        with torch.autocast('cuda', dtype=self.amp_dtype, enabled=self.amp_enabled):
            if self.model.hparams.in_channels != 1:
                target, pred, min_e, loss = self._step_multi_channel_batch(x, weight, meta, phase='Train')
            else:
                target, pred, min_e, loss = self._step_single_channel_batch(x, weight, meta, phase='Train')

        (loss / self.accum_steps).backward()
        self._micro += 1
        stepped = (self._micro % self.accum_steps==0)

        if stepped:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        if stepped and self.global_step % self.log_interval == 0:
            self._calc_metric(target, pred, min_e, phase='Train')
            self.recorder['Train'][f'Train/CurrentLR'] = self.optimizer.param_groups[0]['lr']
            self.recorder['Train'][f'Train/Loss'] = loss.item()
            wandb.log(self.recorder['Train'], step=self.global_step)

        return loss.item(), stepped


    def train(self) -> None:
        self.pbar = tqdm(total=self.num_iterations, desc='Iter')

        while self.global_step < self.num_iterations:
            self.trainset.resample_receptive_field()
            train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                      num_workers=4, pin_memory=True)

            for batch in train_loader:
                loss, stepped = self._train_batch(batch)

                if not stepped:
                    continue

                self.global_step += 1
                self.pbar.update(1)
                self.pbar.set_description(f"Train Loss: {loss:.4f}")

                if self.global_step % self.eval_interval == 0:
                    self.evaluate(self.validset)

                if self.global_step % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.save_dir / f'{self.global_step}_iter.pt')

                if self.global_step >= self.num_iterations:
                    break

        print(f"Best Score: {self.best_score:.4f} at iteration {self.best_iteration}")

        self.pbar.close()


    def evaluate(self, 
                 dataset: Dataset, 
                 external_dataset: bool=False) -> None | float:
        phase = 'Test' if external_dataset else 'Valid'
        self.recorder[f'{phase}'] = {}
        self._eval_samples()

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_bs = self.config.train.get('eval_batch_size', self.batch_size)
        data_loader = DataLoader(dataset, batch_size=eval_bs, shuffle=False)

        total_loss, total_vq_loss, total_recon_loss = 0, 0, 0
        total_ortho_loss = 0 if self.model.quantizer.ortho_reg else None
        total_metrics = [0 for _ in range(len(self.metric))]

        with torch.inference_mode():
            for batch in data_loader:
                x, weight, meta = batch
                x, weight = x.to(self.device), weight.to(self.device)
            
                if self.model.hparams.in_channels != 1:
                    target, pred, min_e, loss = self._step_multi_channel_batch(x, weight, meta, phase=phase)
                else:
                    target, pred, min_e, loss = self._step_single_channel_batch(x, weight, meta, phase=phase)

                self._calc_metric(target, pred, min_e, phase=phase)

                total_loss += loss.item() * x.shape[0]
                total_vq_loss += self.recorder[f'{phase}'][f'{phase}/VQLoss'] * x.shape[0]
                total_recon_loss += self.recorder[f'{phase}'][f'{phase}/ReconLoss'] * x.shape[0]

                if total_ortho_loss is not None:
                    total_ortho_loss += self.recorder[f'{phase}'][f'{phase}/OrthoLoss'] * x.shape[0]

                for i, m in enumerate(self.metric):
                    total_metrics[i] += self.recorder[f'{phase}'][f'{phase}/{m.__class__.__name__}'] * x.shape[0]

        self.recorder[f'{phase}'][f'{phase}/Loss'] = total_loss / len(dataset)
        self.recorder[f'{phase}'][f'{phase}/VQLoss'] = total_vq_loss / len(dataset)
        self.recorder[f'{phase}'][f'{phase}/ReconLoss'] = total_recon_loss / len(dataset)

        if total_ortho_loss is not None:
            self.recorder[f'{phase}'][f'{phase}/OrthoLoss'] = total_ortho_loss / len(dataset)

        for i, m in enumerate(self.metric):
            self.recorder[f'{phase}'][f'{phase}/{m.__class__.__name__}'] = total_metrics[i] / len(dataset)


        if not external_dataset:
            wandb.log(self.recorder[f'{phase}'], step=self.global_step)
            self.pbar.set_description(f"Iter {self.global_step}/{self.num_iterations} | Val Loss: {self.recorder[f'{phase}'][f'{phase}/ReconLoss']:.4f}")
            
            if self.recorder[f'{phase}'][f'{phase}/ReconLoss'] < self.best_score:
                self.best_score = self.recorder[f'{phase}'][f'{phase}/ReconLoss']
                self.best_iteration = self.global_step
                torch.save(self.model.state_dict(), self.save_dir/f'best_model.pt')
                print(f"Iter {self.global_step}, Best Score {self.best_score:.4f}")
                    
        else:
            print(f"Test Loss: {self.recorder[f'{phase}'][f'{phase}/ReconLoss']:.4f}")
            wandb.log(self.recorder[f'{phase}'], step=self.global_step)
            wandb.summary["Test/Loss"] = self.recorder[f'{phase}'][f'{phase}/ReconLoss']

