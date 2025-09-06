"""Run the main code for egg-segmentation-size"""

from pathlib import Path
import logging

import click

from egg_segmentation_size import __version__
from egg_segmentation_size.logging import config_logger
from egg_segmentation_size.segmentor import EggSegmentorTrainer, EggSegmentorInference


logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Shorthand for info/debug/warning/error loglevel (-v/-vv/-vvv/-vvvv)",
)
def egg_segmentation_size_cli(verbose: int) -> None:
    """This repo segments the eggs in images and gives an estimation for their sizes"""
    if verbose == 1:
        log_level = 10
    elif verbose == 2:
        log_level = 20
    elif verbose == 3:
        log_level = 30
    else:
        log_level = 40
    config_logger(log_level)

    click.echo("Run the main code.")


@egg_segmentation_size_cli.command()
@click.option("--img_resize", type=int, default=640, help="Resize images to this size.")
@click.option(
    "--conf_path",
    type=str,
    default="src/egg_segmentation_size/data/data.yaml",
    help="Path to the config file",
)
@click.option(
    "--epochs", type=int, default=100, help="Number of epochs used in training."
)
@click.option("--batch_size", type=int, default=16, help="Batch size used in training.")
@click.option(
    "--device", type=str, default="cuda", help="Use cuda or cpu for training."
)
def train(
    img_resize: int, conf_path: str, epochs: int, batch_size: int, device: str
) -> None:
    """This the CLI for training purposes"""
    segmentation = EggSegmentorTrainer(
        conf=conf_path,
        img_size=img_resize,
        epochs=epochs,
        device=device,
        batch_size=batch_size,
    )
    segmentation.train()
    _ = segmentation.validation()
    segmentation.model_export()


@egg_segmentation_size_cli.command()
@click.option(
    "--model_path",
    type=click.Path(),
    default=Path("./src/egg_segmentation_size/model/egg_segmentor.pt"),
    help="Path to the pre-trained model.",
)
@click.option(
    "--data_path",
    type=click.Path(),
    default=Path("./tests/test_data"),
    help="Path to the test data.",
)
@click.option(
    "--result_path", type=str, default="./results", help="Path to the results."
)
@click.option(
    "--scale_factor",
    type=float,
    default=11.61,
    help="This is the scale factor of the camera to calculate the volume of eggs, scale_factor=DPI/2.54",
)
def infer(
    model_path: Path, data_path: str, result_path: str, scale_factor: float
) -> None:
    """This the CLI for testing purposes"""
    logger.info("Testing the YOLO model for egg detection...")
    inferer = EggSegmentorInference(
        model_path=Path(model_path), result_path=result_path, scale_factor=scale_factor
    )
    segmentations = inferer.inference(data_path=data_path)
    counts = inferer.number_of_eggs(segmentations)
    if counts:
        for key, val in counts.items():
            logger.info(
                "%s eggs are detected in %s as: %s",
                sum(item["count"] for item in val),
                key,
                val,
            )
    res = inferer.results_detail(segmentations)
    if res:
        for key, val in res.items():
            for detection in val:
                logger.info(
                    "In image %s an egg with type: %s, and area(pixel): %d, and volume(cm3) %.2f was detected.",
                    key,
                    detection["class"],
                    detection["areas in pixel"],
                    detection["volume in cm3"],
                )
