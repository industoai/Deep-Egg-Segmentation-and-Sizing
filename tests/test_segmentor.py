"""Test the segmentor module."""

from pathlib import Path
import pytest
from ultralytics import YOLO
from egg_segmentation_size.segmentor import EggSegmentorInference


@pytest.fixture(name="infer_function")
def fixture_infer_function() -> EggSegmentorInference:
    """Fixture to create an EggInference instance before each test."""
    return EggSegmentorInference(
        model_path=Path("./src/egg_segmentation_size/model/egg_segmentor.pt"),
        result_path="",
    )


def test_load_model(infer_function: EggSegmentorInference) -> None:
    """Test the EggInference class."""
    model = infer_function.load_model()
    assert isinstance(model, YOLO)


def test_inference(infer_function: EggSegmentorInference) -> None:
    """Test the inference method of EggInference class."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.jpg")
    assert result


def test_number_of_eggs(infer_function: EggSegmentorInference) -> None:
    """Test the number of eggs detected."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.jpg")
    counts = infer_function.number_of_eggs(result)
    if counts:
        for key, val in counts.items():
            assert sum(item["count"] for item in val) == 3
            assert key == "sample1.jpg"
            assert val == [
                {"class": "White-Egg", "count": 3},
            ]


def test_result_images(infer_function: EggSegmentorInference) -> None:
    """Test the result_images method of EggInference class."""
    result = infer_function.inference(data_path="./tests/test_data/sample1.jpg")
    result_images = infer_function.result_images(result)
    assert result_images
