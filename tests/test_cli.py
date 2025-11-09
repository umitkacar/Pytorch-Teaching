"""Tests for CLI functionality without requiring PyTorch."""

import subprocess
import sys


class TestCLIBasics:
    """Test basic CLI functionality that doesn't require PyTorch."""

    def test_cli_help(self):
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_teaching.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "pytorch-teach" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_version(self):
        """Test that version command works."""
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_teaching.cli", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "2.0.0" in result.stdout

    def test_cli_list_lessons(self):
        """Test that list-lessons command works."""
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_teaching.cli", "list-lessons"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Lesson" in result.stdout

    def test_cli_info(self):
        """Test that info command works."""
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_teaching.cli", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "Python" in result.stdout or "System" in result.stdout


class TestPackageImport:
    """Test that package can be imported."""

    def test_import_package(self):
        """Test that pytorch_teaching can be imported."""
        import pytorch_teaching

        assert hasattr(pytorch_teaching, "__version__")
        assert pytorch_teaching.__version__ == "2.0.0"

    def test_import_cli(self):
        """Test that CLI module can be imported."""
        from pytorch_teaching import cli

        assert hasattr(cli, "app")

    def test_import_lessons_module(self):
        """Test that lessons module can be imported."""
        from pytorch_teaching import lessons  # noqa: F401

        # Just check it imports without error
        assert True
