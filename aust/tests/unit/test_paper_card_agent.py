"""Unit tests for Paper Card Agent."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aust.src.agents.paper_card_agent import PaperCardAgent


class TestPaperCardAgent:
    """Test suite for PaperCardAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent instance."""
        with patch("aust.src.agents.paper_card_agent.ModelFactory"):
            with patch("aust.src.agents.paper_card_agent.ChatAgent"):
                agent = PaperCardAgent(
                    config_path="configs/prompts/paper_card_extraction.yaml",
                    model="openai/gpt-5-nano",
                )
                return agent

    @pytest.fixture
    def sample_metadata(self):
        """Sample paper metadata for testing."""
        return {
            "title": "Test Paper Title",
            "authors": ["Author One", "Author Two"],
            "venue": "Test Conference",
            "year": 2024,
            "arxiv_id": "2401.12345",
            "model_type": "any-to-t",
            "attack_level": "input_level",
            "file_path": "any-to-t/input_level/2401.12345.pdf",
        }

    def test_extract_github_link_found(self, agent):
        """Test GitHub link extraction when link is present."""
        pdf_text = """
        Our code is available at https://github.com/test-org/test-repo for reproducibility.
        See the repository for implementation details.
        """

        github_link = agent.extract_github_link(pdf_text)

        assert github_link == "https://github.com/test-org/test-repo"

    def test_extract_github_link_not_found(self, agent):
        """Test GitHub link extraction when no link is present."""
        pdf_text = "This paper does not have any GitHub repository."

        github_link = agent.extract_github_link(pdf_text)

        assert github_link == "Not provided"

    def test_extract_github_link_multiple_links(self, agent):
        """Test GitHub link extraction when multiple links are present."""
        pdf_text = """
        Code: https://github.com/org1/repo1
        Baseline: https://github.com/org2/repo2
        """

        github_link = agent.extract_github_link(pdf_text)

        # Should return first match
        assert github_link == "https://github.com/org1/repo1"

    def test_extract_github_link_with_dots_in_repo_name(self, agent):
        """Test GitHub link extraction with dots in repository name."""
        pdf_text = "https://github.com/pytorch/vision.git"

        github_link = agent.extract_github_link(pdf_text)

        assert "https://github.com/pytorch/vision" in github_link

    @patch("aust.src.agents.paper_card_agent.UnstructuredIO")
    def test_extract_pdf_text_success_unstructured(self, mock_uio, agent, tmp_path):
        """Test PDF text extraction using UnstructuredIO."""
        # Create temporary PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf content")

        # Mock UnstructuredIO
        mock_instance = Mock()
        mock_instance.parse_file_or_url.return_value = ["Element 1", "Element 2"]
        mock_instance.clean_text_data.return_value = "Cleaned text content"
        mock_uio.return_value = mock_instance

        text = agent.extract_pdf_text(str(test_pdf))

        assert text == "Cleaned text content"
        mock_instance.parse_file_or_url.assert_called_once()
        mock_instance.clean_text_data.assert_called_once()

    @patch("aust.src.agents.paper_card_agent.UnstructuredIO")
    @patch("aust.src.agents.paper_card_agent.create_file_from_raw_bytes")
    def test_extract_pdf_text_fallback_to_base_io(
        self, mock_create_file, mock_uio, agent, tmp_path
    ):
        """Test PDF text extraction fallback when UnstructuredIO fails."""
        # Create temporary PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf content")

        # Mock UnstructuredIO failure
        mock_uio.return_value.parse_file_or_url.side_effect = Exception(
            "UnstructuredIO failed"
        )

        # Mock fallback Base IO
        mock_file_obj = Mock()
        mock_file_obj.docs = [
            {"page_content": "Page 1 content"},
            {"page_content": "Page 2 content"},
        ]
        mock_create_file.return_value = mock_file_obj

        text = agent.extract_pdf_text(str(test_pdf))

        assert "Page 1 content" in text
        assert "Page 2 content" in text
        mock_create_file.assert_called_once()

    def test_extract_pdf_text_file_not_found(self, agent):
        """Test PDF extraction with non-existent file."""
        with pytest.raises(FileNotFoundError):
            agent.extract_pdf_text("/nonexistent/file.pdf")

    def test_validate_card_quality_high_score(self, agent, tmp_path):
        """Test card quality validation with high-quality card."""
        # Create high-quality card
        card_content = """
        # Test Paper

        ## Methodology

        ### Core Method
        This is a detailed description of the core method with more than 50 characters to pass the quality check.

        ### Key Techniques
        - Technique one
        - Technique two
        - Technique three
        - Technique four

        ### Datasets
        MNIST, CIFAR-10, ImageNet datasets were used

        ### Main Findings
        The main findings show that our method achieves state-of-the-art performance across multiple benchmarks with significant improvements over baseline methods. Detailed analysis reveals interesting patterns.

        ### Application to Data-Based Unlearning
        This method can be applied to data-based unlearning by adapting the gradient techniques to test unlearning effectiveness.
        """

        card_path = tmp_path / "test_card.md"
        card_path.write_text(card_content)

        score = agent.validate_card_quality(str(card_path))

        assert score >= 0.7  # Should pass quality threshold
        assert score <= 1.0

    def test_validate_card_quality_low_score(self, agent, tmp_path):
        """Test card quality validation with low-quality card."""
        # Create low-quality card with minimal content
        card_content = """
        # Test Paper

        ## Methodology

        ### Core Method
        Short.

        ### Key Techniques
        - One

        ### Datasets
        Not mentioned

        ### Main Findings
        Brief.
        """

        card_path = tmp_path / "test_card.md"
        card_path.write_text(card_content)

        score = agent.validate_card_quality(str(card_path))

        assert score < 0.7  # Should fail quality threshold

    def test_post_process_card(self, agent, sample_metadata):
        """Test card post-processing."""
        raw_card = """
        # Test Card

        [Current timestamp in ISO format]
        """

        processed_card = agent._post_process_card(
            raw_card, sample_metadata, "https://github.com/test/repo"
        )

        # Check timestamp was replaced
        assert "[Current timestamp in ISO format]" not in processed_card
        assert "**Card Generated**:" in processed_card
        assert "**Agent Model**:" in processed_card

    def test_save_card(self, agent, sample_metadata, tmp_path):
        """Test card saving to file."""
        card_content = "# Test Card Content"

        output_path = agent._save_card(
            card_content, sample_metadata, str(tmp_path)
        )

        # Check file was created
        assert Path(output_path).exists()

        # Check content
        with open(output_path, "r") as f:
            saved_content = f.read()

        assert saved_content == card_content

        # Check path structure matches metadata
        assert "any-to-t/input_level/2401.12345.md" in output_path

    @patch.object(PaperCardAgent, "extract_pdf_text")
    @patch.object(PaperCardAgent, "extract_github_link")
    def test_generate_card_integration(
        self, mock_github, mock_pdf_extract, agent, sample_metadata, tmp_path
    ):
        """Integration test for card generation flow."""
        # Mock PDF extraction
        mock_pdf_extract.return_value = "Extracted PDF text content for testing."

        # Mock GitHub extraction
        mock_github.return_value = "https://github.com/test/repo"

        # Mock LLM response
        mock_response = Mock()
        mock_response.msgs = [Mock(content="# Generated Card Content\n\nTest card.")]
        agent.agent = Mock()
        agent.agent.step.return_value = mock_response

        # Create temporary PDF
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy pdf")

        # Generate card
        card_path = agent.generate_card(
            pdf_path=str(test_pdf),
            metadata=sample_metadata,
            output_dir=str(tmp_path / "cards"),
        )

        # Verify card was created
        assert Path(card_path).exists()

        # Verify methods were called
        mock_pdf_extract.assert_called_once()
        mock_github.assert_called_once()
        agent.agent.step.assert_called_once()


class TestPaperCardAgentErrorHandling:
    """Test error handling in PaperCardAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent instance."""
        with patch("aust.src.agents.paper_card_agent.ModelFactory"):
            with patch("aust.src.agents.paper_card_agent.ChatAgent"):
                agent = PaperCardAgent(
                    config_path="configs/prompts/paper_card_extraction.yaml",
                    model="openai/gpt-5-nano",
                )
                return agent

    @patch.object(PaperCardAgent, "extract_pdf_text")
    def test_generate_card_pdf_extraction_failure(
        self, mock_pdf_extract, agent, tmp_path
    ):
        """Test card generation when PDF extraction fails."""
        # Mock PDF extraction failure
        mock_pdf_extract.side_effect = Exception("PDF extraction failed")

        metadata = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "authors": [],
            "venue": "Test",
            "year": 2024,
            "model_type": "any-to-t",
            "attack_level": "input_level",
            "file_path": "test.pdf",
        }

        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy")

        with pytest.raises(Exception, match="PDF extraction failed"):
            agent.generate_card(
                pdf_path=str(test_pdf), metadata=metadata, output_dir=str(tmp_path)
            )

    @patch.object(PaperCardAgent, "extract_pdf_text")
    @patch.object(PaperCardAgent, "extract_github_link")
    def test_generate_card_llm_extraction_failure(
        self, mock_github, mock_pdf_extract, agent, tmp_path
    ):
        """Test card generation when LLM extraction fails."""
        # Mock PDF extraction success
        mock_pdf_extract.return_value = "PDF text"
        mock_github.return_value = "Not provided"

        # Mock LLM failure
        agent.agent = Mock()
        agent.agent.step.side_effect = Exception("LLM API timeout")

        metadata = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "authors": [],
            "venue": "Test",
            "year": 2024,
            "model_type": "any-to-t",
            "attack_level": "input_level",
            "file_path": "test.pdf",
        }

        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"dummy")

        with pytest.raises(Exception, match="LLM extraction failed"):
            agent.generate_card(
                pdf_path=str(test_pdf), metadata=metadata, output_dir=str(tmp_path)
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
