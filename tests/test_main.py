"""Package level tests"""

from click.testing import CliRunner

from egg_segmentation_size import __version__
from egg_segmentation_size.main import egg_segmentation_size_cli


def test_version() -> None:
    """Unit test for checking the version of the code"""
    assert __version__ == "0.2.0"


def test_egg_segmentation_size_cli() -> None:
    """Unit test for checking the CLI for egg segmentation sizing"""
    runner = CliRunner()
    result = runner.invoke(egg_segmentation_size_cli, ["--help"])
    assert result.exit_code == 0
    assert result


def test_train() -> None:
    """Unit test for training the YOLO model for egg segmentation"""
    runner = CliRunner()
    result = runner.invoke(egg_segmentation_size_cli, ["train", "--help"])
    assert result.exit_code == 0
    assert result


def test_infer() -> None:
    """Unit test for testing the YOLO model for egg segmentation"""
    runner = CliRunner()
    result = runner.invoke(
        egg_segmentation_size_cli,
        ["infer", "--data_path", "./tests/test_data/sample1.jpg", "--result_path", ""],
    )
    assert result.exit_code == 0
    assert result
