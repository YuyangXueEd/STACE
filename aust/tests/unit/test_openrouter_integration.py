"""Integration tests for OpenRouter API with CAMEL-AI (real API calls)."""

import os
import pytest
from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

# Load environment variables
load_dotenv()


class TestOpenRouterRealAPI:
    """Test suite for OpenRouter + CAMEL-AI integration with real API calls."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            pytest.skip("OPENROUTER_API_KEY not found in environment")
        return key

    @pytest.fixture
    def openrouter_model(self, api_key):
        """Create OpenRouter model for testing."""
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5-nano",
            url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model_config_dict={"temperature": 0.5},
        )

    def test_openrouter_model_creation(self, api_key):
        """Test that OpenRouter model is created successfully."""
        # Act
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type="openai/gpt-5-nano",
            url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model_config_dict={"temperature": 0.5},
        )

        # Assert
        assert model is not None
        assert model.model_type == "openai/gpt-5-nano"

    def test_chatagent_with_openrouter(self, openrouter_model):
        """Test ChatAgent initialization with OpenRouter model."""
        # Act
        agent = ChatAgent(
            system_message="You are a helpful assistant.",
            model=openrouter_model
        )

        # Assert
        assert agent is not None
        # ChatAgent is successfully initialized - that's the test

    def test_real_api_call_simple_question(self, openrouter_model):
        """Test actual API call with a simple question."""
        # Arrange
        agent = ChatAgent(
            system_message="You are a helpful assistant that responds concisely.",
            model=openrouter_model
        )

        # Act
        response = agent.step("What is 2+2? Answer with just the number.")
        response_text = response.msgs[0].content

        # Assert
        assert response is not None
        assert response_text is not None
        assert len(response_text) > 0
        assert "4" in response_text

    def test_real_api_call_echo_task(self, openrouter_model):
        """Test API call with echo task."""
        # Arrange
        agent = ChatAgent(
            system_message="You are an echo agent. Respond with: 'Echo: [user message]'",
            model=openrouter_model
        )
        test_message = "Hello CAMEL-AI!"

        # Act
        response = agent.step(test_message)
        response_text = response.msgs[0].content

        # Assert
        assert response is not None
        assert response_text is not None
        assert "Hello CAMEL-AI" in response_text or "echo" in response_text.lower()

    def test_conversation_with_context(self, openrouter_model):
        """Test multi-turn conversation maintains context."""
        # Arrange
        agent = ChatAgent(
            system_message="You are a helpful assistant.",
            model=openrouter_model
        )

        # Act - First message
        response1 = agent.step("My favorite color is blue.")

        # Act - Second message asking about previous context
        response2 = agent.step("What is my favorite color?")
        response2_text = response2.msgs[0].content

        # Assert
        assert response2 is not None
        assert "blue" in response2_text.lower()

    def test_api_key_from_dotenv(self):
        """Test that API key is loaded from .env file."""
        # Act
        api_key = os.getenv("OPENROUTER_API_KEY")

        # Assert
        assert api_key is not None
        assert len(api_key) > 0
        assert api_key.startswith("sk-")
