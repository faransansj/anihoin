"""
Swin Transformer-Tiny 학습 스크립트
- Phase 1: Head만 학습 (5 epoch)
- Phase 2: 전체 fine-tune (낮은 lr)
- WandB 로깅 (선택)
- 최고 val_acc 모델 자동 저장
- Early stopping 지원
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

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
def val_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def test_epoch(model, loader, criterion, device, use_amp, num_classes: int):
    """테스트: 전체 정확도 + 클래스별 정확도"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for imgs, labels in tqdm(loader, desc="  test", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        preds = outputs.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        for pred, label in zip(preds.cpu(), labels.cpu()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    per_class_acc = {
        i: class_correct[i] / class_total[i]
        for i in range(num_classes) if class_total[i] > 0
    }
    return total_loss / total, correct / total, per_class_acc


# ──────────────────────────────────────────────
# 메인 학습
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # WandB 초기화
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("WandB 미설치: uv sync --extra logging 으로 설치 가능")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=args.wandb_run,
        )

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0

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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{args.phase1_epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "phase": 1, "epoch": epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  → best 저장 (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1

    # ────────────────────────────────
    # Phase 2: 전체 fine-tune
    # ────────────────────────────────
    unfreeze_all(model)
    patience_counter = 0  # Phase 2 시작 시 리셋

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.phase2_lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    print(f"\n{'─'*40}")
    print(f"Phase 2 시작 ({args.phase2_epochs} epochs, patience={args.patience})")
    print(f"{'─'*40}")

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{args.phase2_epochs} | "
              f"train_loss: {train_loss:.4f}  train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "phase": 2, "epoch": args.phase1_epochs + epoch,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  → best 저장 (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

    # ────────────────────────────────
    # 최종 테스트
    # ────────────────────────────────
    print(f"\n{'─'*40}")
    print("테스트 세트 평가")
    model.load_state_dict(torch.load(save_dir / "best_model.pth", weights_only=True))
    test_loss, test_acc, per_class_acc = test_epoch(model, test_loader, criterion, device, use_amp, num_classes)
    print(f"  test_loss: {test_loss:.4f}  test_acc: {test_acc:.4f}")

    # 정확도 낮은 클래스 출력 (디버깅용)
    low_acc_classes = sorted(
        [(train_ds.idx_to_class[i], acc) for i, acc in per_class_acc.items()],
        key=lambda x: x[1]
    )[:10]
    print("  [하위 10개 클래스]")
    for cls_name, acc in low_acc_classes:
        print(f"    {cls_name}: {acc:.4f}")
    print(f"{'─'*40}\n")

    if use_wandb:
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb.finish()

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
    parser.add_argument("--data-dir",       default="./dataset/raw")
    parser.add_argument("--save-dir",       default="./checkpoints")
    parser.add_argument("--img-size",       type=int,   default=224)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--phase1-epochs",  type=int,   default=5)
    parser.add_argument("--phase2-epochs",  type=int,   default=30)
    parser.add_argument("--phase2-lr",      type=float, default=1e-5)
    parser.add_argument("--patience",       type=int,   default=7,
                        help="Early stopping patience (0=비활성화)")
    parser.add_argument("--wandb",          action="store_true",
                        help="WandB 로깅 활성화")
    parser.add_argument("--wandb-project",  default="holoscope")
    parser.add_argument("--wandb-run",      default=None)
    args = parser.parse_args()

    train(args)
