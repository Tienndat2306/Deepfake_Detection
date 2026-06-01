"""Trainer class and training loop controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer:
    """
    Quan ly vong lap huan luyen cho DeepfakeDetector bang PyTorch thuan.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion: torch.nn.Module,
        device: torch.device | str,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = torch.device(device)
        self.config = config

        self.model.to(self.device)

        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.patience = int(self.config.get("patience", 5))
        self.min_delta = float(self.config.get("min_delta", 0.0))
        self.checkpoint_path = self.config.get(
            "checkpoint_path", "checkpoints/best_model.pth"
        )
        self.load_best_at_end = bool(self.config.get("load_best_at_end", True))
        self.accumulate_grad_batches = max(
            1, int(self.config.get("accumulate_grad_batches", 1))
        )  # [FIX-BUG-3]
        self.unfreeze_after_epoch = int(
            self.config.get("unfreeze_after_epoch", -1)
        )  # [FIX-BUG-1]
        self.unfreeze_last_n_blocks = max(
            0, int(self.config.get("unfreeze_last_n_blocks", 0))
        )  # [FIX-BUG-1]
        self.unfreeze_backbone_lr = float(
            self.config.get("unfreeze_backbone_lr", self.optimizer.param_groups[0]["lr"])
        )  # [FIX-BUG-1]
        self._unfreeze_applied = False  # [FIX-BUG-1]
        self.save_top_k = max(1, int(self.config.get("save_top_k", 1)))  # [FIX-BUG-4]
        self.saved_checkpoints: List[Dict[str, Any]] = []  # [FIX-BUG-4]
        self.best_checkpoint_path = Path(self.checkpoint_path)  # [FIX-BUG-4]

        # Chi bat AMP khi train tren CUDA.
        self.use_amp = bool(self.config.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_val_auc = float("-inf")
        self.best_epoch = -1
        self.nonfinite_loss_steps = 0
        self.nonfinite_grad_steps = 0

    def _has_finite_gradients(self) -> bool:
        """Kiem tra tat ca gradient hien tai deu finite."""
        for param in self.model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _model_for_ops(self) -> torch.nn.Module:
        """Lay model goc neu dang duoc wrap boi DataParallel/DDP."""
        return self.model.module if hasattr(self.model, "module") else self.model

    def _add_missing_trainable_backbone_params_to_optimizer(self) -> int:
        """
        Sau khi unfreeze backbone, bo sung cac tham so moi trainable vao optimizer.
        """
        model_for_ops = self._model_for_ops()
        backbone = getattr(model_for_ops, "backbone", None)
        if backbone is None:
            return 0

        existing_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group.get("params", [])
        }
        missing_params = [
            param
            for param in backbone.parameters()
            if param.requires_grad and id(param) not in existing_param_ids
        ]
        if not missing_params:
            return 0

        weight_decay = float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
        self.optimizer.add_param_group(
            {
                "params": missing_params,
                "lr": float(self.unfreeze_backbone_lr),
                "weight_decay": weight_decay,
                "name": "backbone_unfrozen",  # [FIX-BUG-1]
            }
        )
        return len(missing_params)

    def _maybe_unfreeze_backbone(self, current_epoch: int) -> None:
        """Mo khoa backbone tai epoch cau hinh."""
        if self._unfreeze_applied:
            return
        if self.unfreeze_after_epoch < 0 or self.unfreeze_last_n_blocks <= 0:
            return
        if int(current_epoch) < int(self.unfreeze_after_epoch):  # [FIX-BUG-1]
            return

        model_for_ops = self._model_for_ops()
        unfreeze_done = False
        if hasattr(model_for_ops, "unfreeze_last_n_blocks"):
            model_for_ops.unfreeze_last_n_blocks(self.unfreeze_last_n_blocks)  # [FIX-BUG-1]
            unfreeze_done = True
        elif hasattr(model_for_ops, "backbone") and hasattr(
            model_for_ops.backbone, "unfreeze_last_n_blocks"
        ):
            model_for_ops.backbone.unfreeze_last_n_blocks(
                self.unfreeze_last_n_blocks
            )  # [FIX-BUG-1]
            unfreeze_done = True
        elif hasattr(model_for_ops, "backbone"):
            # Fallback: mo khoa tham so backbone theo kieu generic.
            for param in model_for_ops.backbone.parameters():
                param.requires_grad = True  # [FIX-BUG-1]
            unfreeze_done = True

        if not unfreeze_done:
            print(
                f"[WARN] Epoch {current_epoch}: khong tim thay ham unfreeze backbone."
            )
            self._unfreeze_applied = True
            return

        added_params = self._add_missing_trainable_backbone_params_to_optimizer()  # [FIX-BUG-1]
        backbone = getattr(model_for_ops, "backbone", None)
        trainable_backbone = 0
        if backbone is not None:
            trainable_backbone = sum(
                int(param.numel()) for param in backbone.parameters() if param.requires_grad
            )
        print(
            f"[FIX-BUG-1] Epoch {current_epoch}: unfreeze last "
            f"{self.unfreeze_last_n_blocks} backbone blocks thanh cong | "
            f"trainable_backbone_params={trainable_backbone:,} | "
            f"optimizer_new_params={added_params}"
        )
        self._unfreeze_applied = True

    @staticmethod
    def _compute_binary_metrics(y_true, y_score) -> Dict[str, float]:
        """Tinh AUC/F1/ACC voi guard cho truong hop y_true 1 class."""
        y_pred = (y_score >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = 0.5
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        return {"auc": auc, "f1": f1, "acc": acc}

    def train_one_epoch(self, current_epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Huan luyen 1 epoch voi AMP + gradient clipping.
        Tra ve {'loss': float, 'acc': float}.
        """
        self.model.train()
        if current_epoch is not None:
            self._maybe_unfreeze_backbone(current_epoch)  # [FIX-BUG-1]

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        # [FIX-BUG-3b] Track so batch da tich luy gradient hop le trong chu ky hien tai.
        _accum_step_count = 0
        for batch_idx, (frames, targets) in enumerate(self.train_loader):
            frames = frames.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True).float().view(-1)
            batch_size = targets.size(0)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(frames).view(-1)  # [B]

            # [FIX C1] Tinh loss ngoai autocast de criterion xu ly o fp32 on dinh hon.
            logits_fp32 = logits.float()
            loss = self.criterion(logits_fp32, targets)

            if not torch.isfinite(loss):
                self.nonfinite_loss_steps += 1
                if self.nonfinite_loss_steps <= 5 or self.nonfinite_loss_steps % 50 == 0:
                    print(
                        f"[WARN] Non-finite loss o train step, bo qua batch. "
                        f"count={self.nonfinite_loss_steps}"
                    )
                # [FIX-BUG-3b] KHONG zero_grad o day: xoa het gradient cua cac batch
                # hop le da tich luy truoc do trong chu ky nay. Chi skip batch nay,
                # giu nguyen gradient da co de chu ky tich luy van chay dung.
                # Tuy nhien phai reset _accum_step_count va zero_grad neu day la batch
                # dau tien cua chu ky (chua co gi de giu lai).
                if _accum_step_count == 0:
                    # Chua tich luy gi, zero_grad an toan.
                    self.optimizer.zero_grad(set_to_none=True)
                # Neu da co grad tich luy (_accum_step_count > 0), khong zero_grad
                # de giu lai gradient tu cac batch hop le truoc do.
                continue

            loss_for_backward = (
                loss / float(self.accumulate_grad_batches)
            )  # [FIX-BUG-3]
            self.scaler.scale(loss_for_backward).backward()  # [FIX-BUG-3]
            _accum_step_count += 1  # [FIX-BUG-3b] dem batch hop le da backward

            # [FIX-BUG-3b] Step khi da du so batch hop le hoac la batch cuoi epoch.
            # Dung _accum_step_count thay vi batch_idx de tranh lech pha khi co skip.
            should_step = (
                (_accum_step_count >= self.accumulate_grad_batches)
                or ((batch_idx + 1) == num_batches and _accum_step_count > 0)
            )  # [FIX-BUG-3b]
            if should_step:
                # Can unscale truoc khi clip gradient:
                # Neu clip tren scaled gradients, nguong clip se bi sai lech
                # (gradient dang bi nhan he so scale), day la loi AMP rat hay gap.
                self.scaler.unscale_(self.optimizer)

                if not self._has_finite_gradients():
                    self.nonfinite_grad_steps += 1
                    if (
                        self.nonfinite_grad_steps <= 5
                        or self.nonfinite_grad_steps % 50 == 0
                    ):
                        print(
                            f"[WARN] Non-finite gradient o train step, bo qua optimizer step. "
                            f"count={self.nonfinite_grad_steps}"
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.update()
                    _accum_step_count = 0  # [FIX-BUG-3b] reset sau khi xoa gradient loi
                    continue

                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                _accum_step_count = 0  # [FIX-BUG-3b] reset dem cho chu ky moi

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                correct = (preds == targets.long()).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += int(correct)
            total_samples += batch_size

        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_correct / max(total_samples, 1)
        return {
            "loss": float(mean_loss),
            "acc": float(mean_acc),
            "skipped_nonfinite_loss": float(self.nonfinite_loss_steps),
            "skipped_nonfinite_grad": float(self.nonfinite_grad_steps),
        }

    def validate(self) -> Dict[str, float]:
        """
        Danh gia tren tap val voi ho tro multi-clip:
        - Loss monitor/early stopping tinh tren tat ca clips.
        - Metrics (AUC/F1/ACC) tinh o cap video sau max-pooling theo clip.
        """
        self.model.eval()

        total_loss_sum = 0.0
        total_clip_samples = 0
        all_video_probs_max = []  # [FIX-BUG-5]
        all_video_probs_mean = []  # [FIX-BUG-5]
        all_video_targets = []

        with torch.inference_mode():
            for frames, targets in self.val_loader:
                targets = targets.to(self.device, non_blocking=True).float().view(-1)
                batch_size = targets.size(0)

                # Ho tro ca batch [B, T, C, H, W] va [B, N, T, C, H, W].
                if frames.ndim == 6:
                    num_clips = int(frames.shape[1])
                    frames_flat = frames.view(
                        batch_size * num_clips,
                        frames.shape[2],
                        frames.shape[3],
                        frames.shape[4],
                        frames.shape[5],
                    )
                elif frames.ndim == 5:
                    num_clips = 1
                    frames_flat = frames
                else:
                    raise ValueError(
                        f"Val batch frames phai co ndim 5 hoac 6, nhan duoc {frames.ndim}."
                    )

                frames_flat = frames_flat.to(self.device, non_blocking=True)
                labels_flat = targets.repeat_interleave(num_clips)

                # Chi dung autocast cho forward pass.
                with autocast("cuda", enabled=self.use_amp):
                    logits_flat = self.model(frames_flat).view(-1)  # [B*num_clips]

                logits_flat = logits_flat.float()
                labels_flat = labels_flat.float()

                # Criterion hien tai tra ve mean loss, scale len tong de monitor ro rang.
                loss_mean = self.criterion(logits_flat, labels_flat)
                loss_sum = loss_mean * labels_flat.numel()
                total_loss_sum += float(loss_sum.item())
                total_clip_samples += int(labels_flat.numel())

                probs = torch.sigmoid(logits_flat).view(batch_size, num_clips)
                video_probs_max, _ = probs.max(dim=1)  # [FIX-BUG-5]
                video_probs_mean = probs.mean(dim=1)  # [FIX-BUG-5]

                all_video_probs_max.append(video_probs_max.cpu())  # [FIX-BUG-5]
                all_video_probs_mean.append(video_probs_mean.cpu())  # [FIX-BUG-5]
                all_video_targets.append(targets.cpu())

        mean_loss = total_loss_sum / max(total_clip_samples, 1)

        if len(all_video_targets) == 0:
            return {
                "loss": float(mean_loss),
                "auc": 0.0,
                "auc_mean": 0.0,
                "auc_max": 0.0,
                "f1": 0.0,
                "f1_mean": 0.0,
                "f1_max": 0.0,
                "acc": 0.0,
                "acc_mean": 0.0,
                "acc_max": 0.0,
            }  # [FIX-BUG-5]

        y_score_max = torch.cat(all_video_probs_max, dim=0).numpy()  # [FIX-BUG-5]
        y_score_mean = torch.cat(all_video_probs_mean, dim=0).numpy()  # [FIX-BUG-5]
        y_true = torch.cat(all_video_targets, dim=0).numpy()
        metrics_max = self._compute_binary_metrics(y_true=y_true, y_score=y_score_max)
        metrics_mean = self._compute_binary_metrics(y_true=y_true, y_score=y_score_mean)

        return {
            "loss": float(mean_loss),
            "auc": float(metrics_mean["auc"]),  # [FIX-BUG-5] metric chinh cho backward compatibility.
            "auc_mean": float(metrics_mean["auc"]),  # [FIX-BUG-5]
            "auc_max": float(metrics_max["auc"]),  # [FIX-BUG-5]
            "f1": float(metrics_mean["f1"]),  # [FIX-BUG-5] metric chinh.
            "f1_mean": float(metrics_mean["f1"]),  # [FIX-BUG-5]
            "f1_max": float(metrics_max["f1"]),  # [FIX-BUG-5]
            "acc": float(metrics_mean["acc"]),  # [FIX-BUG-5] metric chinh.
            "acc_mean": float(metrics_mean["acc"]),  # [FIX-BUG-5]
            "acc_max": float(metrics_max["acc"]),  # [FIX-BUG-5]
        }

    def _save_best_checkpoint(self, epoch: int, val_auc: float) -> None:
        """Luu checkpoint theo che do top-k."""
        ckpt_base = Path(self.checkpoint_path)
        ckpt_base.parent.mkdir(parents=True, exist_ok=True)
        if self.save_top_k > 1:
            score_tag = f"{float(val_auc):.6f}".replace(".", "_")
            ckpt_path = ckpt_base.with_name(
                f"{ckpt_base.stem}_epoch{int(epoch):03d}_auc{score_tag}{ckpt_base.suffix}"
            )  # [FIX-BUG-4]
        else:
            ckpt_path = ckpt_base

        model_for_ckpt = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_for_ckpt, "save_checkpoint"):
            model_for_ckpt.save_checkpoint(str(ckpt_path), epoch=epoch, val_auc=val_auc)
        else:
            payload = {
                "epoch": int(epoch),
                "val_auc": float(val_auc),
                "model_state_dict": model_for_ckpt.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
            }
            torch.save(payload, ckpt_path)

        self.saved_checkpoints = [
            entry
            for entry in self.saved_checkpoints
            if Path(entry["path"]) != ckpt_path
        ]  # [FIX-BUG-4]
        self.saved_checkpoints.append(
            {"path": str(ckpt_path), "score": float(val_auc), "epoch": int(epoch)}
        )  # [FIX-BUG-4]
        self.saved_checkpoints.sort(key=lambda x: float(x["score"]), reverse=True)  # [FIX-BUG-4]
        self.best_checkpoint_path = Path(self.saved_checkpoints[0]["path"])  # [FIX-BUG-4]

        while len(self.saved_checkpoints) > self.save_top_k:  # [FIX-BUG-4]
            worst = self.saved_checkpoints.pop(-1)
            worst_path = Path(worst["path"])
            if worst_path.exists() and worst_path != self.best_checkpoint_path:
                worst_path.unlink()

    def _load_best_checkpoint_into_model(self) -> bool:
        """
        Nap best checkpoint vao model sau khi fit (optional).
        """
        ckpt_path = self.best_checkpoint_path if self.best_checkpoint_path.exists() else Path(self.checkpoint_path)  # [FIX-BUG-4]
        if not ckpt_path.exists():
            return False

        model_for_load = self.model.module if hasattr(self.model, "module") else self.model
        try:
            if hasattr(model_for_load, "load_checkpoint"):
                model_for_load.load_checkpoint(str(ckpt_path))
                return True

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            if not isinstance(checkpoint, dict):
                return False
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            if not isinstance(state_dict, dict):
                return False
            model_for_load.load_state_dict(state_dict, strict=False)
            return True
        except Exception as exc:
            print(f"[WARN] Khong load duoc best checkpoint vao cuoi training: {exc}")
            return False

    def fit(self, num_epochs: int, start_epoch: int = 1) -> Dict[str, float]:
        """
        Chay vong lap train/val nhieu epoch voi early stopping.
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs phai > 0.")
        if start_epoch <= 0:
            raise ValueError("start_epoch phai > 0.")

        epochs_no_improve = 0

        end_epoch = start_epoch + num_epochs - 1
        for epoch in range(start_epoch, end_epoch + 1):
            train_metrics = self.train_one_epoch(current_epoch=epoch)
            val_metrics = self.validate()

            # Update scheduler sau khi co val metric.
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["auc_mean"])  # [FIX-BUG-5]
                else:
                    self.scheduler.step()

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch}/{end_epoch} | "  # [FIX-BUG-1]
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auc_mean={val_metrics['auc_mean']:.4f} | "  # [FIX-BUG-5]
                f"val_auc_max={val_metrics['auc_max']:.4f} | "  # [FIX-BUG-5]
                f"val_f1_mean={val_metrics.get('f1_mean', 0.0):.4f} | "  # [FIX-BUG-5]
                f"val_f1_max={val_metrics.get('f1_max', 0.0):.4f} | "  # [FIX-BUG-5]
                f"val_acc_mean={val_metrics.get('acc_mean', 0.0):.4f} | "  # [FIX-BUG-5]
                f"val_acc_max={val_metrics.get('acc_max', 0.0):.4f} | "  # [FIX-BUG-5]
                f"lr={current_lr:.6e}"
            )

            monitor_auc = float(val_metrics["auc_mean"])  # [FIX-BUG-5]
            # Chi luu checkpoint khi AUC co y nghia (tot hon random baseline).
            if monitor_auc > (self.best_val_auc + self.min_delta) and monitor_auc > 0.5:
                self.best_val_auc = monitor_auc
                self.best_epoch = epoch
                epochs_no_improve = 0
                self._save_best_checkpoint(epoch=epoch, val_auc=self.best_val_auc)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"Early stopping tai epoch {epoch}. "
                        f"Best val_auc={self.best_val_auc:.4f} tai epoch {self.best_epoch}."
                    )
                    break

        if self.load_best_at_end and self.best_epoch >= 1:
            loaded = self._load_best_checkpoint_into_model()
            if loaded:
                print(
                    f"Loaded best checkpoint sau khi fit: {self.best_checkpoint_path} "  # [FIX-BUG-4]
                    f"(epoch={self.best_epoch}, val_auc={self.best_val_auc:.4f})."
                )

        return {"best_val_auc": float(self.best_val_auc), "best_epoch": int(self.best_epoch)}

