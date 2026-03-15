"""
Swin Transformer-Tiny 학습 스크립트
- Phase 1: Head만 학습 (5 epoch)
- Phase 2: 전체 fine-tune (낮은 lr)
- WandB 로깅 (선택)
- 최고 val_acc 모델 자동 저장
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from tqdm import tqdm

from dataset import build_dataloaders

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def freeze_backbone(model: nn.Module):
    """head만 학습 (Phase 1)"""
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1: head만 학습 | 학습 파라미터: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """전체 학습 (Phase 2)"""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 2: 전체 fine-tune | 학습 파라미터: {trainable:,}")


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# 메인 학습
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터로더
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        root_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(train_ds.classes)
    print(f"클래스 수: {num_classes}")

    # 클래스 맵 저장
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_ds.save_class_map(save_dir / "class_map.json")

    # 모델
    model = build_model(num_classes).to(device)

    # Loss: Label Smoothing으로 과적합 방지
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0.0

    # ────────────────────────────────
    # Phase 1: Head만 학습
    # ────────────────────────────────
    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_epochs)

    print(f"\n{'─'*40}")
    print(f"Phase 1 시작 ({args.phase1_epochs} epochs)")
    print(f"{'─'*40}")

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{args.phase1_epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  → best 저장 (val_acc: {val_acc:.4f})")

    # ────────────────────────────────
    # Phase 2: 전체 fine-tune
    # ────────────────────────────────
    unfreeze_all(model)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    print(f"\n{'─'*40}")
    print(f"Phase 2 시작 ({args.phase2_epochs} epochs)")
    print(f"{'─'*40}")

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{args.phase2_epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  → best 저장 (val_acc: {val_acc:.4f})")

    # ────────────────────────────────
    # 최종 테스트
    # ────────────────────────────────
    print(f"\n{'─'*40}")
    print("테스트 세트 평가")
    model.load_state_dict(torch.load(save_dir / "best_model.pth"))
    test_loss, test_acc = val_epoch(model, test_loader, criterion, device)
    print(f"  test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")
    print(f"{'─'*40}\n")

    # 학습 설정 저장
    config = {
        "num_classes": num_classes,
        "img_size": args.img_size,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "model": "swin_tiny_patch4_window7_224",
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"학습 완료! 최고 val_acc: {best_val_acc:.4f}")
    print(f"모델 저장 위치: {save_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       default="../dataset/raw")
    parser.add_argument("--save-dir",       default="../checkpoints")
    parser.add_argument("--img-size",       type=int,   default=224)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--phase1-epochs",  type=int,   default=5)
    parser.add_argument("--phase2-epochs",  type=int,   default=30)
    parser.add_argument("--phase2-lr",      type=float, default=1e-5)
    args = parser.parse_args()

    train(args)
