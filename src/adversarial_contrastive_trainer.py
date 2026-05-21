import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contrastive_trainer import ContrastiveTrainer
from domain_adaptation import dann_lambda
from util.utils import AverageMeter, adjust_learning_rate, warmup_learning_rate


class AdversarialContrastiveTrainer(ContrastiveTrainer):
    """
    Supervised contrastive trainer with DANN-style source/target feature alignment.

    Source batches use labels for SupCon. Target batches only contribute domain labels,
    which fits unlabeled target H&E crops as long as they can be loaded by the existing
    crop pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        domain_discriminator: nn.Module,
        target_loader: DataLoader,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer,
        num_classes: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        save_dir: str = "./contrastive_checkpoints",
        log_interval: int = 10,
        args=None,
        config=None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_classes=num_classes,
            scheduler=scheduler,
            device=device,
            save_dir=save_dir,
            log_interval=log_interval,
            args=args,
            config=config,
        )
        self.domain_discriminator = domain_discriminator.to(device)
        self.target_loader = target_loader
        self.target_iter = iter(target_loader)

        adv_config = (config or {}).get("adversarial", {})
        self.domain_loss_weight = adv_config.get("domain_loss_weight", 1.0)
        self.max_domain_lambda = adv_config.get("domain_lambda", 1.0)
        self.domain_warmup_epochs = adv_config.get("domain_warmup_epochs", 5)
        self.source_domain_grad = adv_config.get("source_domain_grad", True)
        self.target_domain_grad = adv_config.get("target_domain_grad", True)
        self.max_epochs = (config or {}).get("epoch_max", 100)

        self.history.update({
            "train_supcon_loss": [],
            "train_domain_loss": [],
            "train_domain_acc": [],
            "domain_lambda": [],
        })

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "domain_discriminator_state_dict": self.domain_discriminator.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "domain_discriminator_state_dict" in checkpoint:
            self.domain_discriminator.load_state_dict(checkpoint["domain_discriminator_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", 110.0)
        self.history = checkpoint.get("history", self.history)

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    def _batch_to_two_view_tensor(self, batch):
        images = batch["image"]
        labels = batch["label"]

        masks = batch.get("mask", None)
        if masks is not None:
            images[0] = torch.cat([images[0], masks[0]], dim=1)
            images[1] = torch.cat([images[1], masks[1]], dim=1)

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        return images, labels

    def _next_target_batch(self):
        try:
            return next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)
            return next(self.target_iter)

    def train_epoch(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        self.domain_discriminator.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        supcon_losses = AverageMeter()
        domain_losses = AverageMeter()
        domain_accs = AverageMeter()
        lambda_meter = AverageMeter()

        end = time.time()
        for idx, source_batch in enumerate(train_loader):
            data_time.update(time.time() - end)
            target_batch = self._next_target_batch()

            source_images, source_labels = self._batch_to_two_view_tensor(source_batch)
            target_images, _ = self._batch_to_two_view_tensor(target_batch)
            bsz = source_labels.shape[0]

            warmup_learning_rate(self.args, epoch, idx, len(train_loader), optimizer)
            lambda_value = dann_lambda(
                epoch=epoch,
                batch_idx=idx,
                batches_per_epoch=len(train_loader),
                max_epochs=self.max_epochs,
                max_lambda=self.max_domain_lambda,
                warmup_epochs=self.domain_warmup_epochs,
            )

            source_features, _, source_projection = model(source_images)
            target_features, _, _ = model(target_images)

            f1, f2 = torch.split(source_projection, [bsz, bsz], dim=0)
            source_projection = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            supcon_loss = criterion(source_projection, source_labels)

            source_domain_features = source_features if self.source_domain_grad else source_features.detach()
            target_domain_features = target_features if self.target_domain_grad else target_features.detach()
            domain_features = torch.cat([source_domain_features, target_domain_features], dim=0)
            source_domain = torch.zeros(source_features.size(0), dtype=torch.long, device=self.device)
            target_domain = torch.ones(target_features.size(0), dtype=torch.long, device=self.device)
            domain_labels = torch.cat([source_domain, target_domain], dim=0)

            domain_logits = self.domain_discriminator(domain_features, lambda_value=lambda_value)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            loss = supcon_loss + self.domain_loss_weight * domain_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean().item()

            losses.update(loss.item(), bsz)
            supcon_losses.update(supcon_loss.item(), bsz)
            domain_losses.update(domain_loss.item(), bsz)
            domain_accs.update(domain_acc, bsz)
            lambda_meter.update(lambda_value, bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % self.log_interval == 0:
                print(
                    "Train: [{0}][{1}/{2}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                    "supcon {supcon.val:.3f} ({supcon.avg:.3f})\t"
                    "domain {domain.val:.3f} ({domain.avg:.3f})\t"
                    "domain_acc {acc.val:.3f} ({acc.avg:.3f})\t"
                    "lambda {lam.val:.3f}".format(
                        epoch,
                        idx + 1,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        supcon=supcon_losses,
                        domain=domain_losses,
                        acc=domain_accs,
                        lam=lambda_meter,
                    )
                )
                sys.stdout.flush()

        self.history["train_supcon_loss"].append(supcon_losses.avg)
        self.history["train_domain_loss"].append(domain_losses.avg)
        self.history["train_domain_acc"].append(domain_accs.avg)
        self.history["domain_lambda"].append(lambda_meter.avg)
        return losses.avg

    def save_history(self):
        history_path = os.path.join(self.save_dir, "training_history.json")
        json_history = {}
        for key, values in self.history.items():
            json_history[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]

        import json
        with open(history_path, "w") as f:
            json.dump(json_history, f, indent=2)
