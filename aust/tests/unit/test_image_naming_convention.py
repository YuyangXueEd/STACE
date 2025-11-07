"""
Unit tests for AC5: Standardized Image Naming Convention (Story 5.2).

Tests cover:
1. sanitize_prompt_for_filename() helper function
2. Filename validation in CodeSynthesizer
3. Filename pattern matching
4. Edge cases and error handling
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from aust.src.data_models.code_synthesis import sanitize_prompt_for_filename


class TestSanitizePromptForFilename:
    """Test AC5: sanitize_prompt_for_filename() helper function."""

    def test_basic_sanitization(self) -> None:
        """Test basic prompt sanitization with spaces and punctuation."""
        result = sanitize_prompt_for_filename("A photograph of a red apple")
        assert result == "a_photograph_of_a_red_apple"

    def test_remove_special_characters(self) -> None:
        """Test that special characters are removed."""
        result = sanitize_prompt_for_filename("Van Gogh style portrait!!!")
        assert result == "van_gogh_style_portrait"

    def test_long_prompt_truncation(self) -> None:
        """Test that long prompts are truncated to max_length."""
        long_prompt = "Very " * 30  # 150 chars
        result = sanitize_prompt_for_filename(long_prompt, max_length=50)
        assert len(result) <= 50
        assert result.endswith("very")  # Should not end with underscore
        assert not result.endswith("_")

    def test_only_special_characters_fallback(self) -> None:
        """Test fallback to 'generated' when only special chars remain."""
        result = sanitize_prompt_for_filename("###@@@!!!")
        assert result == "generated"

    def test_empty_string_fallback(self) -> None:
        """Test fallback to 'generated' for empty input."""
        result = sanitize_prompt_for_filename("")
        assert result == "generated"

    def test_collapse_multiple_spaces(self) -> None:
        """Test that multiple spaces are collapsed to single underscore."""
        result = sanitize_prompt_for_filename("photo    of     apple")
        assert result == "photo_of_apple"
        assert "__" not in result

    def test_hyphen_to_underscore(self) -> None:
        """Test that hyphens are converted to underscores."""
        result = sanitize_prompt_for_filename("cinematic-portrait-style")
        assert result == "cinematic_portrait_style"
        assert "-" not in result

    def test_mixed_case_to_lowercase(self) -> None:
        """Test that mixed case is converted to lowercase."""
        result = sanitize_prompt_for_filename("Van Gogh STYLE Portrait")
        assert result == "van_gogh_style_portrait"
        assert result.islower()

    def test_numeric_characters_preserved(self) -> None:
        """Test that numbers are preserved in the slug."""
        result = sanitize_prompt_for_filename("Generate 5 images of apples")
        assert result == "generate_5_images_of_apples"
        assert "5" in result

    def test_custom_max_length(self) -> None:
        """Test that custom max_length parameter works."""
        result = sanitize_prompt_for_filename("A very long prompt text", max_length=10)
        assert len(result) <= 10
        assert result == "a_very_lon"

    def test_leading_trailing_underscores_stripped(self) -> None:
        """Test that leading/trailing underscores are stripped."""
        result = sanitize_prompt_for_filename("___photo of apple___")
        assert result == "photo_of_apple"
        assert not result.startswith("_")
        assert not result.endswith("_")


class TestFilenamePatternValidation:
    """Test AC5: Filename pattern validation regex."""

    @pytest.fixture
    def filename_pattern(self) -> re.Pattern:
        """Provide the filename validation pattern."""
        return re.compile(r"^[a-z0-9_]+_\d+_[a-f0-9]{8}\.png$")

    def test_valid_filename_matches(self, filename_pattern: re.Pattern) -> None:
        """Test that valid filenames match the pattern."""
        valid_filenames = [
            "photograph_of_apple_42_a3f7b9c2.png",
            "cinematic_portrait_123_f8e4d1a0.png",
            "dog_running_456_b2c9a7e3.png",
            "generated_0_ffffffff.png",
            "a_1_00000000.png",
        ]
        for filename in valid_filenames:
            assert filename_pattern.match(filename), f"Expected {filename} to match"

    def test_invalid_uppercase_slug(self, filename_pattern: re.Pattern) -> None:
        """Test that uppercase characters in slug are rejected."""
        assert not filename_pattern.match("Photo_Of_Apple_42_a3f7b9c2.png")

    def test_invalid_special_chars_in_slug(self, filename_pattern: re.Pattern) -> None:
        """Test that special characters in slug are rejected."""
        assert not filename_pattern.match("photo-of-apple_42_a3f7b9c2.png")
        assert not filename_pattern.match("photo.of.apple_42_a3f7b9c2.png")

    def test_invalid_non_integer_seed(self, filename_pattern: re.Pattern) -> None:
        """Test that non-integer seeds are rejected."""
        assert not filename_pattern.match("photograph_of_apple_42.5_a3f7b9c2.png")
        assert not filename_pattern.match("photograph_of_apple_abc_a3f7b9c2.png")

    def test_invalid_uuid_length(self, filename_pattern: re.Pattern) -> None:
        """Test that UUIDs not exactly 8 hex chars are rejected."""
        assert not filename_pattern.match("photograph_of_apple_42_a3f7b9c.png")  # 7 chars
        assert not filename_pattern.match("photograph_of_apple_42_a3f7b9c22.png")  # 9 chars

    def test_invalid_uppercase_uuid(self, filename_pattern: re.Pattern) -> None:
        """Test that uppercase hex in UUID is rejected."""
        assert not filename_pattern.match("photograph_of_apple_42_A3F7B9C2.png")

    def test_invalid_non_hex_uuid(self, filename_pattern: re.Pattern) -> None:
        """Test that non-hex characters in UUID are rejected."""
        assert not filename_pattern.match("photograph_of_apple_42_g3f7b9c2.png")

    def test_invalid_file_extension(self, filename_pattern: re.Pattern) -> None:
        """Test that non-.png extensions are rejected."""
        assert not filename_pattern.match("photograph_of_apple_42_a3f7b9c2.jpg")
        assert not filename_pattern.match("photograph_of_apple_42_a3f7b9c2.PNG")

    def test_invalid_missing_components(self, filename_pattern: re.Pattern) -> None:
        """Test that filenames missing required components are rejected."""
        assert not filename_pattern.match("photograph_of_apple_42.png")  # Missing UUID
        assert not filename_pattern.match("photograph_of_apple_a3f7b9c2.png")  # Missing seed
        assert not filename_pattern.match("42_a3f7b9c2.png")  # Missing prompt slug


class TestCodeSynthesizerFilenameValidation:
    """Test AC5: Filename validation in CodeSynthesizer._validate_generation_outputs()."""

    def test_valid_filenames_pass_validation(self, tmp_path: Path) -> None:
        """Test that valid filenames pass validation."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create valid images
        image1 = output_dir / "photograph_of_apple_42_a3f7b9c2.png"
        image2 = output_dir / "portrait_of_church_123_f8e4d1a0.png"
        image1.write_bytes(b"fake image data")
        image2.write_bytes(b"fake image data")

        # Create manifest
        manifest = [
            {
                "image_path": str(image1),
                "prompt": "photograph of apple",
                "prompt_slug": "photograph_of_apple",
                "seed": 42,
                "seed_value": 42,
                "uuid_value": "a3f7b9c2",
            },
            {
                "image_path": str(image2),
                "prompt": "portrait of church",
                "prompt_slug": "portrait_of_church",
                "seed": 123,
                "seed_value": 123,
                "uuid_value": "f8e4d1a0",
            },
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert count == 2
        assert error is None

    def test_invalid_filename_format_fails_validation(self, tmp_path: Path) -> None:
        """Test that invalid filename format triggers validation error."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create image with invalid filename (missing uuid)
        invalid_image = output_dir / "photograph_of_apple_42.png"
        invalid_image.write_bytes(b"fake image data")

        # Create manifest
        manifest = [
            {
                "image_path": str(invalid_image),
                "prompt": "photograph of apple",
                "prompt_slug": "photograph_of_apple",
                "seed": 42,
                "seed_value": 42,
                "uuid_value": "a3f7b9c2",
            }
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "do not follow naming convention" in error
        assert "prompt_slug_seed_uuid.png" in error

    def test_uppercase_in_slug_fails_validation(self, tmp_path: Path) -> None:
        """Test that uppercase characters in slug fail validation."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create image with uppercase in slug
        invalid_image = output_dir / "Photograph_Of_Apple_42_a3f7b9c2.png"
        invalid_image.write_bytes(b"fake image data")

        manifest = [
            {
                "image_path": str(invalid_image),
                "prompt": "Photograph Of Apple",
                "prompt_slug": "photograph_of_apple",
                "seed": 42,
                "seed_value": 42,
                "uuid_value": "a3f7b9c2",
            }
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "do not follow naming convention" in error

    def test_special_chars_in_slug_fails_validation(self, tmp_path: Path) -> None:
        """Test that special characters in slug fail validation."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create image with special chars in slug
        invalid_image = output_dir / "photo-of-apple!_42_a3f7b9c2.png"
        invalid_image.write_bytes(b"fake image data")

        manifest = [
            {
                "image_path": str(invalid_image),
                "prompt": "photo-of-apple!",
                "prompt_slug": "photo_of_apple",
                "seed": 42,
                "seed_value": 42,
                "uuid_value": "a3f7b9c2",
            }
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "do not follow naming convention" in error

    def test_missing_metadata_fails_validation(self, tmp_path: Path) -> None:
        """Test that missing manifest metadata triggers validation error."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        image = output_dir / "photograph_of_apple_42_a3f7b9c2.png"
        image.write_bytes(b"fake image data")

        manifest = [
            {
                "image_path": str(image),
                "prompt": "photograph of apple",
                # Intentionally omit metadata keys
                "seed": 42,
            }
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "missing required metadata" in error

    def test_metadata_mismatch_fails_validation(self, tmp_path: Path) -> None:
        """Test that mismatched metadata triggers validation error."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        image = output_dir / "photograph_of_apple_42_a3f7b9c2.png"
        image.write_bytes(b"fake image data")

        manifest = [
            {
                "image_path": str(image),
                "prompt": "photograph of apple",
                "prompt_slug": "photograph_of_banana",
                "seed": 42,
                "seed_value": 99,
                "uuid_value": "ffffffff",
            }
        ]
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "Manifest metadata does not match filenames" in error

    def test_multiple_invalid_filenames_reported(self, tmp_path: Path) -> None:
        """Test that multiple invalid filenames are reported (up to 3)."""
        from aust.src.agents.code_synthesizer import CodeSynthesizerAgent

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Create 4 images with invalid filenames
        for i in range(4):
            invalid_image = output_dir / f"image_{i}.png"  # Wrong format
            invalid_image.write_bytes(b"fake image data")

        manifest = []
        for i in range(4):
            manifest.append(
                {
                    "image_path": str(output_dir / f"image_{i}.png"),
                    "prompt": f"prompt {i}",
                    "prompt_slug": sanitize_prompt_for_filename(f"prompt {i}"),
                    "seed": i,
                    "seed_value": i,
                    "uuid_value": "ffffffff",
                }
            )
        (output_dir / "generation_manifest.json").write_text(json.dumps(manifest))

        synthesizer = CodeSynthesizerAgent()
        count, error = synthesizer._validate_generation_outputs(output_dir)

        assert error is not None
        assert "(4 total)" in error  # Should indicate 4 total invalid
        assert "image_0.png" in error  # Should show first 3
        assert "image_1.png" in error
        assert "image_2.png" in error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
