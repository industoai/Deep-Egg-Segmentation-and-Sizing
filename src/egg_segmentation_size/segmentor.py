"""This is the code for training the YOLO model for egg segmentation."""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Mapping, List

from collections import Counter
from ultralytics import YOLO
import numpy as np
import cv2


logger = logging.getLogger(__name__)


@dataclass
class EggSegmentorTrainer:
    """Class for training the YOLO model for egg segmentation."""

    conf: str = field(default="src/egg_segmentation_size/data/data.yaml")
    epochs: int = field(default=100)
    img_size: int = field(default=640)
    batch_size: int = field(default=16)
    device: str = field(default="cuda")
    model: Any = field(init=False)

    def train(self) -> None:
        """Train the YOLO model for egg segmentation."""
        logger.info("Start training the YOLO model for egg segmentation.")
        self.model = YOLO("yolov8n-seg.pt")
        self.model.train(
            data=self.conf,
            epochs=self.epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
        )

    def validation(self) -> Any:
        """Validate the YOLO model for egg segmentation."""
        logger.info("Validating the YOLO model for egg segmentation.")
        return self.model.val()

    def model_export(self) -> None:
        """Export the YOLO model for egg segmentation."""
        logger.info("Exporting the YOLO model for egg segmentation.")
        self.model.export(format="onnx")


@dataclass
class EggSegmentorInference:
    """Class for testing the YOLO model for egg segmentation."""

    model_path: Optional[Any] = field(default=None)
    result_path: Optional[str] = field(default=None)
    scale_factor: float = field(default=11.61)

    def __post_init__(self) -> None:
        """Post-initialization method for EggSegmentorInference."""
        if self.model_path is None or not self.model_path.exists():
            raise ValueError("Model does not exist or the path is not correct.")

    def load_model(self) -> Any:
        """Load the YOLO model for egg detection."""
        logger.info("Loading the trained model for egg segmentation.")
        return YOLO(self.model_path)

    def inference(self, data_path: str) -> Any:
        """Inference code for egg segmentation"""
        if not Path(data_path).exists():
            logger.error("Data path does not exist or the path is not correct.")
        model = self.load_model()
        results = model(
            data_path,
            save=False if not self.result_path else True,  # pylint: disable=R1719
            project=self.result_path,
            name="detections",
        )
        return results

    @staticmethod
    def _shoelace_area(polygon: Any) -> float:
        """Calculate the area of a polygon using the shoelace formula."""
        x, y = polygon[:, 0], polygon[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    @staticmethod
    def number_of_eggs(detections: Any) -> Mapping[str, Any]:
        """Count the number of eggs detected."""
        counts = {}
        for result in detections:
            class_count = Counter(int(box.cls.item()) for box in result.boxes)
            temp = []
            for name, count in class_count.items():
                temp.append({"class": result.names[name], "count": count})
            file_name = Path(result.path).name
            counts[str(file_name)] = temp
        return counts

    def _egg_volume(self, polygon: Any, circularity_thr: int = 15) -> float:
        """Calculate the volume of eggs based on the detected polygon for each egg."""
        polygon = polygon.reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(polygon)  # pylint: disable=E1101
        minor_axis, major_axis = (
            ellipse[1][0] / self.scale_factor,
            ellipse[1][1] / self.scale_factor,
        )

        if (major_axis - minor_axis) > circularity_thr:
            return 4 * np.pi * (major_axis / 2) * ((minor_axis / 2) ** 2) / 3000
        return 4 * np.pi * (((major_axis + minor_axis) / 4) ** 3) / 3000

    def results_detail(self, detections: Any) -> Mapping[str, Any]:
        """Get the detailed results of the segmented eggs such as bounding boxes, class names, and confidences."""
        results = {}
        for result in detections:
            temp = []
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks.xy
                for i, mask in enumerate(masks):
                    polygon = np.array(mask, dtype=np.float32)
                    temp.append(
                        {
                            "class": result.names[int(boxes.cls[i].item())],
                            "confidence": boxes.conf[i].item(),
                            "areas in pixel": self._shoelace_area(polygon),
                            "volume in cm3": self._egg_volume(polygon),
                        }
                    )
                file_name = Path(result.path).name
                results[str(file_name)] = temp
        return results

    @staticmethod
    def result_images(detections: Any) -> List[Any]:
        """Make a list of the result images with detections."""
        images = []
        for result in detections:
            images.append(np.array(result.plot())[:, :, [2, 1, 0]])
        return images
