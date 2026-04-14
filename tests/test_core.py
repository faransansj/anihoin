import pytest
import numpy as np
import torch
from pathlib import Path
import shutil
from PIL import Image
from unittest.mock import MagicMock, patch

from dataset import HoloDataset, get_train_transforms, get_val_transforms
from crawling.danbooru_crawler import DanbooruCrawler

# ──────────────────────────────────────────────
# Dataset Tests
# ──────────────────────────────────────────────


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """테스트용 가상 데이터셋 구조 생성"""
    classes = ["char_a", "char_b"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(cls_dir / f"img_{i}.jpg")
    return tmp_path


def test_holo_dataset_init(mock_dataset_dir):
    """클래스 자동 생성 및 샘플 수집 검증"""
    ds = HoloDataset(mock_dataset_dir, split="train", val_ratio=0, test_ratio=0)
    assert len(ds.classes) == 2
    assert "char_a" in ds.classes
    assert "char_b" in ds.classes
    assert len(ds.samples) == 20


def test_holo_dataset_split(mock_dataset_dir):
    """데이터 분할 비율 검증"""
    # 20장 중 10% val(2장), 10% test(2장) -> train 16장
    train_ds = HoloDataset(
        mock_dataset_dir, split="train", val_ratio=0.1, test_ratio=0.1
    )
    val_ds = HoloDataset(mock_dataset_dir, split="val", val_ratio=0.1, test_ratio=0.1)
    test_ds = HoloDataset(mock_dataset_dir, split="test", val_ratio=0.1, test_ratio=0.1)

    assert len(train_ds) == 16
    assert len(val_ds) == 2
    assert len(test_ds) == 2


def test_transforms_output_shape():
    """트랜스폼 결과 텐서 크기 검증 (224x224)"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)

    train_res = train_tf(image=img)["image"]
    val_res = val_tf(image=img)["image"]

    assert train_res.shape == (3, 224, 224)
    assert val_res.shape == (3, 224, 224)


# ──────────────────────────────────────────────
# Crawler Tests
# ──────────────────────────────────────────────


@patch("requests.Session.get")
def test_crawler_get_posts(mock_get, tmp_path):
    """API 응답 파싱 로직 검증"""
    # 가짜 API 응답 설정
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 1, "file_url": "http://test.com/1.jpg", "rating": "g"},
        {
            "id": 2,
            "file_url": "http://test.com/2.jpg",
            "rating": "x",
        },  # SFW 필터링 대상
    ]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    crawler = DanbooruCrawler(output_dir=tmp_path)
    posts = crawler._get_posts("test_tag", 1)

    assert len(posts) == 2
    assert posts[0]["id"] == 1


@patch("requests.Session.get")
def test_crawler_collect_queue(mock_get, tmp_path):
    """큐 수집 시 SFW 필터링 및 확장자 검증"""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 1, "file_url": "http://test.com/1.jpg", "rating": "g"},  # OK
        {"id": 2, "file_url": "http://test.com/2.png", "rating": "s"},  # OK
        {"id": 3, "file_url": "http://test.com/3.txt", "rating": "g"},  # Invalid Ext
        {"id": 4, "file_url": "http://test.com/4.jpg", "rating": "x"},  # Not SFW
    ]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    crawler = DanbooruCrawler(output_dir=tmp_path)
    char_dir = tmp_path / "test_char"
    char_dir.mkdir()

    queue = crawler._collect_queue("test_tag", char_dir, 10)

    # 1, 2번만 통과해야 함
    assert len(queue) == 2
    assert "1.jpg" in queue[0][1].name
    assert "2.png" in queue[1][1].name
