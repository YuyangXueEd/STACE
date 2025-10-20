"""Integration tests for paper card generation workflow.

These tests use real files and LLM calls (mocked for CI/CD).
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock

from aust.src.agents.paper_card_agent import PaperCardAgent
from scripts.generate_paper_cards import PaperCardBatchProcessor


class TestPaperCardGenerationIntegration:
    """Integration tests for end-to-end paper card generation."""

    @pytest.fixture
    def test_data_dir(self, tmp_path):
        """Create test data directory structure."""
        papers_dir = tmp_path / ".papers"
        papers_dir.mkdir()

        # Create category structure
        (papers_dir / "any-to-t" / "input_level").mkdir(parents=True)

        # Create sample PDF (dummy content)
        test_pdf = papers_dir / "any-to-t" / "input_level" / "2401.12345.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 dummy content for testing")

        # Create paper metadata
        metadata = [
            {
                "title": "Test Paper for Integration Testing",
                "authors": ["Test Author One", "Test Author Two"],
                "venue": "Test Conference",
                "year": 2024,
                "arxiv_id": "2401.12345",
                "model_type": "any-to-t",
                "attack_level": "input_level",
                "file_path": "any-to-t/input_level/2401.12345.pdf",
            }
        ]

        metadata_file = papers_dir / "paper_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

        return {
            "papers_dir": papers_dir,
            "cards_dir": tmp_path / ".paper_cards",
            "metadata_file": metadata_file,
            "test_pdf": test_pdf,
            "metadata": metadata[0],
        }

    @pytest.mark.skipif(
        True, reason="Requires real LLM API - run manually for validation"
    )
    def test_generate_single_card_real_llm(self, test_data_dir):
        """Test generating a single paper card with real LLM (manual test)."""
        # Initialize agent
        agent = PaperCardAgent(
            config_path="configs/prompts/paper_card_extraction.yaml",
            model="openai/gpt-5-nano",
        )

        # Generate card
        card_path = agent.generate_card(
            pdf_path=str(test_data_dir["test_pdf"]),
            metadata=test_data_dir["metadata"],
            output_dir=str(test_data_dir["cards_dir"]),
        )

        # Verify card exists
        assert Path(card_path).exists()

        # Verify card structure
        with open(card_path, "r") as f:
            card_content = f.read()

        # Check required sections
        assert "# Test Paper for Integration Testing" in card_content
        assert "## Metadata" in card_content
        assert "## Methodology" in card_content
        assert "## Experiment Design" in card_content
        assert "## Key Results (Summary)" in card_content
        assert "## Relevance to Our Work" in card_content

        # Validate quality
        quality_score = agent.validate_card_quality(card_path)
        print(f"Card quality score: {quality_score:.2f}")

        assert quality_score > 0.0  # Should have some quality score

    @patch("aust.src.agents.paper_card_agent.ChatAgent")
    @patch("aust.src.agents.paper_card_agent.ModelFactory")
    @patch.object(PaperCardAgent, "extract_pdf_text")
    def test_generate_single_card_mocked_llm(
        self, mock_extract_pdf, mock_model_factory, mock_chat_agent, test_data_dir
    ):
        """Test generating a single paper card with mocked LLM."""
        # Mock PDF extraction
        mock_extract_pdf.return_value = """
        Test Paper Content

        Abstract
        This paper presents a novel approach to adversarial attacks on multimodal models.

        Methodology
        We use gradient-based optimization with cross-modal attention manipulation.

        Experiments
        We tested on GPT-4V, LLaVA, and VideoLLaMA using AdvBench dataset.

        Results
        Our method achieves 85% attack success rate with SSIM=0.92.

        GitHub: https://github.com/test/test-repo
        """

        # Mock LLM response
        mock_response = Mock()
        mock_response.msgs = [
            Mock(
                content="""# Test Paper for Integration Testing

## Metadata
- **Authors**: Test Author One, Test Author Two
- **Venue**: Test Conference
- **Year**: 2024
- **ArXiv ID**: 2401.12345
- **Model Type**: any-to-t
- **Attack Level**: input_level
- **GitHub**: https://github.com/test/test-repo

## Quick Summary
This paper presents a novel gradient-based approach to adversarial attacks on multimodal models.

## Methodology

### Core Method
Gradient-based optimization with cross-modal attention manipulation for adversarial attack generation.

### Key Techniques
- Gradient-based perturbation generation
- Cross-modal attention manipulation
- Adversarial optimization

### Algorithm/Approach
1. Extract features from multimodal model
2. Optimize perturbations using gradient descent
3. Evaluate attack success rate

## Experiment Design

### Datasets
AdvBench dataset

### Models Tested
GPT-4V, LLaVA, VideoLLaMA

### Evaluation Metrics
Attack Success Rate (ASR), SSIM

### Baselines
Standard adversarial attacks

## Key Results (Summary)

### Main Findings
Our method achieves 85% attack success rate with high perceptual quality (SSIM=0.92).

### Performance Highlights
85% ASR on GPT-4V, 90% on LLaVA

### Limitations
Limited to specific model architectures

## Implementation Details

### Hyperparameters (if mentioned)
Not provided in paper

### Computational Requirements (if mentioned)
Not provided in paper

## Relevance to Our Work

### Application to Data-Based Unlearning
This gradient-based approach could be adapted to test unlearning robustness.

### Application to Concept Erasure
Cross-modal attention manipulation could test concept erasure effectiveness.

### Potential Attack Methods
1. Gradient-based membership inference
2. Cross-modal data reconstruction

## Key Quotes
> "Our method achieves state-of-the-art attack success rates."

## Related Work Mentioned
- Visual Adversarial Examples (2023)
- Multimodal Jailbreaking (2024)

## Citation
```bibtex
@article{2401.12345,
  title={Test Paper for Integration Testing},
  author={Test Author One, Test Author Two},
  journal={Test Conference},
  year={2024},
  arxiv={2401.12345}
}
```

---

**Card Generated**: 2024-10-20T12:00:00Z
**Agent Model**: openai/gpt-5-nano
"""
            )
        ]

        mock_agent_instance = Mock()
        mock_agent_instance.step.return_value = mock_response
        mock_chat_agent.return_value = mock_agent_instance

        # Initialize agent
        agent = PaperCardAgent(
            config_path="configs/prompts/paper_card_extraction.yaml",
            model="openai/gpt-5-nano",
        )

        # Generate card
        card_path = agent.generate_card(
            pdf_path=str(test_data_dir["test_pdf"]),
            metadata=test_data_dir["metadata"],
            output_dir=str(test_data_dir["cards_dir"]),
        )

        # Verify card exists
        assert Path(card_path).exists()

        # Verify card content
        with open(card_path, "r") as f:
            card_content = f.read()

        assert "# Test Paper for Integration Testing" in card_content
        assert "## Methodology" in card_content
        assert "https://github.com/test/test-repo" in card_content

        # Validate quality
        quality_score = agent.validate_card_quality(card_path)
        assert quality_score >= 0.7  # Should pass quality threshold

    @patch("aust.src.agents.paper_card_agent.ChatAgent")
    @patch("aust.src.agents.paper_card_agent.ModelFactory")
    @patch.object(PaperCardAgent, "extract_pdf_text")
    def test_batch_processing_single_paper(
        self, mock_extract_pdf, mock_model_factory, mock_chat_agent, test_data_dir
    ):
        """Test batch processing with single paper."""
        # Mock PDF extraction
        mock_extract_pdf.return_value = "Test PDF content for batch processing"

        # Mock LLM response
        mock_response = Mock()
        mock_response.msgs = [
            Mock(
                content="""# Test Card

## Metadata
- **Authors**: Test Authors
- **Venue**: Test Venue
- **Year**: 2024

## Methodology

### Core Method
Test method with sufficient length to pass quality checks for validation purposes.

### Key Techniques
- Technique 1
- Technique 2
- Technique 3

### Datasets
Test datasets: MNIST, CIFAR-10

### Main Findings
Test findings with sufficient length to meet the quality threshold requirements for validation.

### Application to Data-Based Unlearning
Test relevance analysis with sufficient length to pass quality checks.
"""
            )
        ]

        mock_agent_instance = Mock()
        mock_agent_instance.step.return_value = mock_response
        mock_chat_agent.return_value = mock_agent_instance

        # Initialize batch processor
        processor = PaperCardBatchProcessor(
            papers_dir=str(test_data_dir["papers_dir"]),
            cards_dir=str(test_data_dir["cards_dir"]),
            metadata_file=str(test_data_dir["metadata_file"]),
            config_path="configs/prompts/paper_card_extraction.yaml",
            model="openai/gpt-5-nano",
        )

        # Process batch
        processor.process_batch()

        # Verify card was generated
        expected_card_path = (
            test_data_dir["cards_dir"]
            / "any-to-t"
            / "input_level"
            / "2401.12345.md"
        )
        assert expected_card_path.exists()

        # Verify progress tracking
        progress_file = test_data_dir["cards_dir"] / ".generation_progress.json"
        assert progress_file.exists()

        with open(progress_file, "r") as f:
            progress = json.load(f)

        assert "2401.12345" in progress["processed"]
        assert progress["completed"] == 1

        # Verify card metadata
        card_metadata_file = test_data_dir["cards_dir"] / "card_metadata.json"
        assert card_metadata_file.exists()

        with open(card_metadata_file, "r") as f:
            card_metadata = json.load(f)

        assert len(card_metadata) == 1
        assert card_metadata[0]["arxiv_id"] == "2401.12345"
        assert card_metadata[0]["processing_status"] == "success"

    def test_batch_processor_resume_from_checkpoint(self, test_data_dir):
        """Test batch processor resume functionality."""
        # Create existing progress file
        progress_file = test_data_dir["cards_dir"] / ".generation_progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        existing_progress = {
            "processed": ["2401.12345"],
            "failed": [],
            "total": 1,
            "completed": 1,
        }

        progress_file.write_text(json.dumps(existing_progress, indent=2))

        # Initialize processor
        processor = PaperCardBatchProcessor(
            papers_dir=str(test_data_dir["papers_dir"]),
            cards_dir=str(test_data_dir["cards_dir"]),
            metadata_file=str(test_data_dir["metadata_file"]),
        )

        # Load progress
        progress = processor.load_progress()

        assert "2401.12345" in progress["processed"]
        assert progress["completed"] == 1

    def test_batch_processor_failed_extraction_logging(self, test_data_dir):
        """Test failed extraction logging."""
        # Initialize processor
        processor = PaperCardBatchProcessor(
            papers_dir=str(test_data_dir["papers_dir"]),
            cards_dir=str(test_data_dir["cards_dir"]),
            metadata_file=str(test_data_dir["metadata_file"]),
        )

        # Log failed extraction
        failed_entry = {
            "arxiv_id": "2401.12345",
            "pdf_path": str(test_data_dir["test_pdf"]),
            "error_type": "PDFExtractionError",
            "error_message": "Test error",
            "timestamp": "2024-10-20T12:00:00Z",
            "retry_count": 3,
        }

        processor.save_failed_extraction(failed_entry)

        # Verify failed extractions file
        failed_file = test_data_dir["cards_dir"] / ".failed_extractions.json"
        assert failed_file.exists()

        with open(failed_file, "r") as f:
            failed_extractions = json.load(f)

        assert len(failed_extractions) == 1
        assert failed_extractions[0]["arxiv_id"] == "2401.12345"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
