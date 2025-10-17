"""ChatAgent example with basic interaction logging using CAUST logging framework."""

import os
from pathlib import Path
from dotenv import load_dotenv

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from aust.src.logging_config import setup_logging, get_logger

# Setup logging using CAUST framework
setup_logging(log_level="INFO", log_dir=Path("aust/logs"), enable_console=True, enable_file=True)
logger = get_logger(__name__)

load_dotenv()

# Create OpenRouter model
logger.info("Creating OpenRouter model: openai/gpt-5-nano")
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="openai/gpt-5-nano",
    url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_config_dict={"temperature": 0.5},
)
logger.info("Model created successfully")

# Create ChatAgent
sys_msg = "You are a helpful assistant."
logger.info(f"Creating ChatAgent with system message: {sys_msg}")
camel_agent = ChatAgent(
    system_message=sys_msg,
    model=model,
)
logger.info("ChatAgent created successfully")

# Test interaction with logging
user_msg = "Tell me about quantum computing in a simple term."
logger.info(f"User input: {user_msg}")

response = camel_agent.step(user_msg)
response_content = response.msgs[0].content

logger.info(f"Agent response received - length: {len(response_content)} characters")
logger.debug(f"Full response: {response_content}")

print("\n" + "=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response_content)
print("=" * 60)
