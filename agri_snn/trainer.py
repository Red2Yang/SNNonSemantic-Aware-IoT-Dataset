import torch
import torch.nn as nn
import os
from datetime import datetime
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .network import DualBranchAgriSNN
import logging
import tensorflow as tf

logging.getLogger('spikingjelly').setLevel(logging.ERROR)


class SingleTaskTrainer:
    def __init__(self, model: DualBranchAgriSNN, encoder,
                 device='cuda', lr=1e-3,
                 class_weights: torch.Tensor = None,
                 log_dir: str = 'runs'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.encoder = encoder

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
            if class_weights is not None else None
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

        # ── TensorBoard ──
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = Path("runs") / run_name
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        ##self.writer = SummaryWriter(log_dir=str(log_dir))
        #print(f"[TensorBoard] 日志保存至 {log_dir}")

        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [],  'test_acc': []
        }
        self.global_step = 0
        self.label_encoders = None      # 延时绑定

    def safe_reset(self):
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def set_label_encoders(self, label_encoders):
        self.label_encoders = label_encoders

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            x_ms, x_rgb, ms_mask, rgb_mask, y_action = batch
            x_ms = x_ms.to(self.device)
            x_rgb = x_rgb.to(self.device)
            ms_mask = ms_mask.to(self.device)
            rgb_mask = rgb_mask.to(self.device)
            y_action = y_action.to(self.device)

            ms_seq = self.encoder(x_ms, ms_mask)
            rgb_seq = self.encoder(x_rgb, rgb_mask)
            action_rate = self.model(ms_seq, rgb_seq)

            loss = self.criterion(action_rate, y_action)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.safe_reset()

            total_loss += loss.item() * x_ms.size(0)
            pred = action_rate.argmax(dim=1)
            correct += (pred == y_action).sum().item()
            total += x_ms.size(0)
            all_preds.extend(pred.cpu().tolist())

            #self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{correct/total:.3f}'})

        avg_loss = total_loss / total
        acc = correct / total

        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(acc)

        #self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        #self.writer.add_scalar('train/epoch_accuracy', acc, epoch)
        #self.writer.add_scalar('train/learning_rate',
        #                       self.optimizer.param_groups[0]['lr'], epoch)

        # 预测分布（可选）
        if self.label_encoders is not None:
            class_names = self.label_encoders['Action_Suggested'].classes_
            dist = Counter(all_preds)
            readable = {class_names[int(k)]: v for k, v in dist.items()}
            #self.writer.add_text('train/pred_distribution', str(readable), epoch)
            print(f"  预测分布: {readable}")
        else:
            dist = Counter(all_preds)
            print(f"  预测分布: {dict(sorted(dist.items()))}")

        self.scheduler.step(acc)
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, test_loader, epoch, phase='test'):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch in test_loader:
            x_ms, x_rgb, ms_mask, rgb_mask, y_action = batch
            x_ms = x_ms.to(self.device)
            x_rgb = x_rgb.to(self.device)
            ms_mask = ms_mask.to(self.device)
            rgb_mask = rgb_mask.to(self.device)
            y_action = y_action.to(self.device)

            ms_seq = self.encoder(x_ms, ms_mask)
            rgb_seq = self.encoder(x_rgb, rgb_mask)
            action_rate = self.model(ms_seq, rgb_seq)
            loss = self.criterion(action_rate, y_action)

            total_loss += loss.item() * x_ms.size(0)
            correct += (action_rate.argmax(dim=1) == y_action).sum().item()
            total += x_ms.size(0)
            self.safe_reset()

        avg_loss = total_loss / total
        acc = correct / total
        self.history[f'{phase}_loss'].append(avg_loss)
        self.history[f'{phase}_acc'].append(acc)

        #self.writer.add_scalar(f'{phase}/epoch_loss', avg_loss, epoch)
        #self.writer.add_scalar(f'{phase}/epoch_accuracy', acc, epoch)
        return avg_loss, acc

    def fit(self, train_loader, test_loader, epochs=150, patience=30):
        best_acc = 0
        patience_counter = 0
        best_epoch = 0
        print("=" * 60)
        print("  开始训练（单任务 + 类别权重 + 融合分支）")
        print("=" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            test_loss, test_acc = self.evaluate(test_loader, epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                best_epoch = epoch
                torch.save(self.model.state_dict(), 'best_agri_snn.pth')
                print(f"  ★ 新最佳模型 (Acc: {best_acc:.3f})")
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"[Epoch {epoch}/{epochs}]")
                print(f"  Train - Loss:{train_loss:.4f}  Acc:{train_acc:.3f}")
                print(f"  Test  - Loss:{test_loss:.4f}   Acc:{test_acc:.3f}")
                print(f"  Best Acc: {best_acc:.3f} (Epoch {best_epoch})  "
                      f"Patience: {patience_counter}/{patience}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= patience:
                print(f"\n[早停] 连续 {patience} 轮无改善，停止训练")
                break

        #self.writer.close()
        print("=" * 60)
        print(f"  训练结束，最佳测试准确率: {best_acc:.3f}")
        print("=" * 60)
        return self.history