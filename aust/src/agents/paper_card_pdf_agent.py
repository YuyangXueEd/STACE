"""
Paper Card Agent that streams PDFs directly to the OpenRouter Responses API.

This agent avoids up-front PDF text extraction and instead lets the model
consume the PDF via OpenAI's responses endpoint with `input_file` attachments.
"""

import base64
import json
import os
import re
import time
import yaml
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from aust.src.utils.logging_config import setup_logging, get_logger
from aust.src.utils.model_config import load_model_settings

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "prompts" / "paper_card_pdf_extraction.yaml"
DEFAULT_FALLBACK_CONFIG = PACKAGE_ROOT / "configs" / "prompts" / "paper_card_extraction.yaml"

_PAPER_CARD_PDF_MODEL_FALLBACK = {
    "model_name": "openai/gpt-5-nano",
    "config": {
        "temperature": 0.3,
        "max_tokens": 4000,
        "top_p": 0.9,
    },
}

load_dotenv()

# Ensure logging matches the rest of the project.
setup_logging(
    log_level="INFO",
    log_dir=Path("./logs"),
    enable_console=True,
    enable_file=True,
)
logger = get_logger(__name__)


class PaperCardPdfAgent:
    """Agent for generating structured paper cards using PDF inputs."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        if config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
        else:
            provided_path = Path(config_path)
            if not provided_path.is_absolute():
                provided_path = provided_path.resolve()
            self.config_path = provided_path
        self.config = self._load_config()
        model_settings = load_model_settings("paper_card_pdf_agent", _PAPER_CARD_PDF_MODEL_FALLBACK, logger=logger)

        if model:
            self.config["model"] = model
        if temperature is not None:
            self.config["temperature"] = temperature

        self.config.setdefault("model", model_settings["model_name"])
        for key, value in model_settings.get("config", {}).items():
            self.config.setdefault(key, value)

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. Cannot initialize PaperCardPdfAgent."
            )

        self.api_key = api_key
        self.base_url = self.config.get(
            "api_base_url", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.default_headers = self._default_headers()
        self.max_upload_bytes = int(self.config.get("max_upload_bytes", 20_000_000))
        fallback_config_value = self.config.get("fallback_text_config")
        if fallback_config_value:
            fallback_path = Path(fallback_config_value)
            if not fallback_path.is_absolute():
                project_candidate = (PROJECT_ROOT / fallback_path).resolve()
                package_candidate = (PACKAGE_ROOT / fallback_path).resolve()
                if project_candidate.exists():
                    fallback_path = project_candidate
                elif package_candidate.exists():
                    fallback_path = package_candidate
                else:
                    fallback_path = project_candidate
            self.fallback_config_path = fallback_path
        else:
            self.fallback_config_path = DEFAULT_FALLBACK_CONFIG
        self.pdf_compression_config = self.config.get("pdf_compression", {})
        self._fallback_agent = None

        logger.info(
            "Initialized PaperCardPdfAgent with model %s", self.config["model"]
        )

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as handle:
            return yaml.safe_load(handle)

    def _default_headers(self) -> Dict[str, str]:
        """Build default headers for OpenRouter (optional but recommended)."""
        headers: Dict[str, str] = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_APP_TITLE")
        if title:
            headers["X-Title"] = title
        return headers

    def _build_user_prompt(self, metadata: Dict[str, Any]) -> str:
        """Fill the user prompt template with metadata."""
        prompt = self.config["user_prompt_template"]
        for key, value in metadata.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def generate_card(
        self,
        pdf_path: str,
        metadata: Dict[str, Any],
        output_dir: str = ".paper_cards/",
    ) -> str:
        """Generate a paper card by streaming the PDF directly to the API."""
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Augment metadata with fields referenced in the prompt template.
        prompt_metadata = {
            "title": metadata.get("title", "Unknown Title"),
            "authors": ", ".join(metadata.get("authors", [])) or "Unknown Authors",
            "venue": metadata.get("venue", "Unknown Venue"),
            "year": metadata.get("year", "Unknown Year"),
            "arxiv_id": metadata.get("arxiv_id", "Unknown ID"),
            "model_type": metadata.get("model_type", "Unknown"),
            "attack_level": metadata.get("attack_level", "Unknown"),
            "file_path": metadata.get("file_path", ""),
        }

        user_prompt = self._build_user_prompt(prompt_metadata)

        pdf_cleanup_paths: List[Path] = []
        pdf_path_for_upload = pdf_path_obj
        file_size = pdf_path_for_upload.stat().st_size

        if self.max_upload_bytes and file_size > self.max_upload_bytes:
            logger.warning(
                "PDF size (%d bytes) exceeds upload limit (%d); attempting compression",
                file_size,
                self.max_upload_bytes,
            )
            compressed_pdf = self._compress_pdf(pdf_path_obj)
            if compressed_pdf:
                pdf_cleanup_paths.append(compressed_pdf)
                pdf_path_for_upload = compressed_pdf
                file_size = pdf_path_for_upload.stat().st_size
                logger.info(
                    "Compressed PDF for %s now %d bytes",
                    prompt_metadata["arxiv_id"],
                    file_size,
                )

        if self.max_upload_bytes and file_size > self.max_upload_bytes:
            logger.warning(
                "PDF remains above upload limit after compression; falling back to text extraction"
            )
            self._cleanup_local_pdfs(pdf_cleanup_paths)
            return self._fallback_generate(
                pdf_path=pdf_path,
                metadata=metadata,
                output_dir=output_dir,
            )

        pdf_data_url = self._encode_pdf_to_data_url(pdf_path_for_upload)
        plugins = self.config.get("plugins")

        max_tokens_config = self.config.get("max_output_tokens")
        adaptive_tokens = max_tokens_config is not None
        if adaptive_tokens:
            current_max_tokens = int(max_tokens_config)
            max_tokens_limit = int(
                self.config.get("max_output_tokens_limit", current_max_tokens)
            )
            growth_factor = float(
                self.config.get("max_output_tokens_growth_factor", 1.5)
            )
        else:
            current_max_tokens = None
            max_tokens_limit = None  # unused
            growth_factor = None  # unused

        # Basic retry loop mirroring the text-based agent behaviour.
        retry_cfg = self.config.get("retry_config", {})
        max_retries = int(retry_cfg.get("max_retries", 3))
        delay = float(retry_cfg.get("initial_delay", 1.0))
        backoff = float(retry_cfg.get("backoff_factor", 2.0))

        last_exception: Optional[Exception] = None

        try:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        "Running PDF extraction for %s (attempt %d/%d)",
                        prompt_metadata["arxiv_id"],
                        attempt,
                        max_retries,
                    )

                    while True:
                        response_json = self._call_chat_completion(
                            system_prompt=self.config["system_prompt"],
                            user_prompt=user_prompt,
                            pdf_filename=pdf_path_for_upload.name,
                            pdf_data_url=pdf_data_url,
                            plugins=plugins,
                            max_tokens=current_max_tokens,
                        )

                        finish_reason = self._extract_finish_reason(response_json)
                        card_content = self._extract_chat_content(response_json)

                        if not card_content:
                            logger.error(
                                "Empty response content for %s (finish_reason=%s, max_tokens=%s): %s",
                                prompt_metadata["arxiv_id"],
                                finish_reason,
                                str(current_max_tokens),
                                json.dumps(response_json)[:2000],
                            )

                            if (
                                adaptive_tokens
                                and finish_reason in {"length", "max_output_tokens"}
                                and current_max_tokens < max_tokens_limit
                            ):
                                current_max_tokens = min(
                                    int(current_max_tokens * growth_factor),
                                    max_tokens_limit,
                                )
                                logger.warning(
                                    "Increasing max tokens to %d for %s due to truncated output",
                                    current_max_tokens,
                                    prompt_metadata["arxiv_id"],
                                )
                                continue

                            raise ValueError("Model returned empty response for paper card generation")

                        if (
                            adaptive_tokens
                            and finish_reason in {"length", "max_output_tokens"}
                            and current_max_tokens < max_tokens_limit
                        ):
                            current_max_tokens = min(
                                int(current_max_tokens * growth_factor),
                                max_tokens_limit,
                            )
                            logger.warning(
                                "Response for %s truncated (finish_reason=%s); retrying with max_tokens=%d",
                                prompt_metadata["arxiv_id"],
                                finish_reason,
                                current_max_tokens,
                            )
                            continue

                        if finish_reason in {"length", "max_output_tokens"} and not adaptive_tokens:
                            logger.warning(
                                "Response for %s truncated (finish_reason=%s) but no max_tokens limit configured; proceeding with available content",
                                prompt_metadata["arxiv_id"],
                                finish_reason,
                            )

                        logger.info(
                            "PDF extraction successful for %s, received %d characters",
                            prompt_metadata["arxiv_id"],
                            len(card_content),
                        )

                        return self._finalize_card(
                            card_content=card_content,
                            metadata=metadata,
                            output_dir=output_dir,
                        )

                except Exception as exc:
                    logger.error(
                        "PDF extraction failed for %s on attempt %d/%d: %s",
                        prompt_metadata["arxiv_id"],
                        attempt,
                        max_retries,
                        exc,
                    )
                    last_exception = exc
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff

            raise RuntimeError(
                f"Failed to generate card for {prompt_metadata['arxiv_id']} after "
                f"{max_retries} attempts"
            ) from last_exception
        finally:
            self._cleanup_local_pdfs(pdf_cleanup_paths)

    def _finalize_card(
        self,
        card_content: str,
        metadata: Dict[str, Any],
        output_dir: str,
    ) -> str:
        """Post-process, ensure placeholders are filled, and persist the card."""
        processed = self._post_process_card(card_content)
        output_path = self._save_card(processed, metadata, output_dir)
        return output_path

    def _post_process_card(self, card_content: str) -> str:
        """Ensure the markdown card contains required metadata footers."""
        generation_date = datetime.now(timezone.utc).isoformat()

        if "**Card Generated**:" not in card_content:
            card_content += (
                f"\n\n---\n\n**Card Generated**: {generation_date}\n"
                f"**Agent Model**: {self.config['model']}\n"
            )

        # If GitHub field is left empty, normalise it to "Not provided".
        github_pattern = r"\*\*GitHub\*\*:\s*$"
        if re.search(github_pattern, card_content, re.MULTILINE):
            card_content = re.sub(
                github_pattern,
                "**GitHub**: Not provided",
                card_content,
                flags=re.MULTILINE,
            )

        return card_content

    def _save_card(
        self,
        card_content: str,
        metadata: Dict[str, Any],
        output_dir: str,
    ) -> str:
        arxiv_id = metadata.get("arxiv_id", "unknown_id")
        file_path = metadata.get("file_path", f"{arxiv_id}.pdf")
        output_path = Path(output_dir) / file_path.replace(".pdf", ".md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as handle:
            handle.write(card_content)

        logger.info("Paper card saved to %s", output_path)
        return str(output_path)

    def validate_card_quality(self, card_path: str) -> float:
        """Reuse the same heuristic quality scoring as the text-based agent."""
        with open(card_path, "r") as handle:
            card_content = handle.read()

        score = 0.0
        thresholds = self.config.get("quality_thresholds", {})

        core_method = self._extract_section(card_content, "### Core Method")
        if core_method and len(core_method) > thresholds.get("min_core_method_length", 50):
            score += 0.3

        key_techniques = self._extract_section(card_content, "### Key Techniques")
        if key_techniques:
            technique_count = len(
                [line for line in key_techniques.splitlines() if line.strip().startswith("- ")]
            )
            if technique_count >= thresholds.get("min_key_techniques", 3):
                score += 0.2

        datasets = self._extract_section(card_content, "### Datasets")
        if datasets and len(datasets) > 10 and "Not provided" not in datasets:
            score += 0.2

        findings = self._extract_section(card_content, "### Main Findings")
        if findings and len(findings) > thresholds.get("min_main_findings_length", 100):
            score += 0.2

        relevance = self._extract_section(
            card_content, "### Application to Data-Based Unlearning"
        )
        if relevance and len(relevance) > thresholds.get("min_relevance_length", 50):
            score += 0.1

        logger.info("Card quality score for %s: %.2f", card_path, score)
        return score

    @staticmethod
    def _extract_section(card_content: str, heading: str) -> Optional[str]:
        """Helper to pull a section between a heading and the next heading."""
        pattern = rf"{re.escape(heading)}\n(.+?)(?=\n###|\n##|$)"
        match = re.search(pattern, card_content, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _cleanup_local_pdfs(paths: List[Path]):
        """Delete any temporary PDFs created during compression."""
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning("Failed to remove temporary PDF %s: %s", path, exc)

    def _compress_pdf(self, pdf_path: Path) -> Optional[Path]:
        """Attempt to compress the PDF using configured strategies."""
        config = self.pdf_compression_config or {}
        if not config.get("enabled", True):
            logger.info("PDF compression disabled via configuration")
            return None

        method = str(config.get("method", "ghostscript")).lower()
        min_ratio = float(config.get("min_ratio", 0.98))

        methods_to_try: List[str]
        if method == "auto":
            methods_to_try = ["ghostscript", "python"]
        else:
            methods_to_try = [method]

        for method_name in methods_to_try:
            if method_name == "python":
                compressed = self._compress_pdf_python(
                    pdf_path, config.get("python", {}), min_ratio
                )
            elif method_name == "ghostscript":
                compressed = self._compress_pdf_ghostscript(
                    pdf_path, config.get("ghostscript", {}), min_ratio
                )
            else:
                logger.warning("Unknown compression method '%s'", method_name)
                continue

            if compressed:
                return compressed

        logger.info(
            "No configured PDF compression method succeeded; using original PDF"
        )
        return None

    def _compress_pdf_python(
        self,
        pdf_path: Path,
        method_config: Dict[str, Any],
        min_ratio: float,
    ) -> Optional[Path]:
        """Compress PDF using pure-Python tooling (pikepdf)."""
        try:
            import pikepdf
        except ImportError:
            logger.warning(
                "pikepdf not installed; cannot perform python-based PDF compression"
            )
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        compression_name = method_config.get("compression_level", "default")
        compression_kwargs: Dict[str, Any] = {}

        compression_enum = getattr(pikepdf, "CompressionLevel", None)
        if compression_enum is not None:
            default_compression = getattr(compression_enum, "default", None)
            compression_level = getattr(
                compression_enum,
                compression_name,
                default_compression,
            )
            if compression_level is not None:
                compression_kwargs["compression"] = compression_level
        else:
            logger.debug(
                "pikepdf.CompressionLevel not available; using library defaults"
            )

        object_stream_name = method_config.get("object_stream_mode", "generate")
        object_stream_enum = getattr(pikepdf, "ObjectStreamMode", None)
        if object_stream_enum is not None:
            default_object_mode = getattr(object_stream_enum, "generate", None)
            object_stream_mode = getattr(
                object_stream_enum,
                object_stream_name,
                default_object_mode,
            )
            if object_stream_mode is not None:
                compression_kwargs["object_stream_mode"] = object_stream_mode
        else:
            logger.debug(
                "pikepdf.ObjectStreamMode not available; skipping object stream parameter"
            )

        stream_encode_name = method_config.get("stream_encode_level", "default")
        stream_encode_enum = getattr(pikepdf, "StreamEncodeLevel", None)
        if stream_encode_enum is not None:
            default_stream_encode = getattr(stream_encode_enum, "default", None)
            stream_encode_level = getattr(
                stream_encode_enum,
                stream_encode_name,
                default_stream_encode,
            )
            if stream_encode_level is not None:
                compression_kwargs["stream_encode_level"] = stream_encode_level
        else:
            logger.debug(
                "pikepdf.StreamEncodeLevel not available; skipping stream encode parameter"
            )

        linearize = bool(method_config.get("linearize", True))

        logger.info("Compressing PDF %s using pikepdf", pdf_path.name)
        try:
            with pikepdf.open(str(pdf_path)) as pdf:
                pdf.save(
                    tmp_path,
                    linearize=linearize,
                    **compression_kwargs,
                )
        except Exception as exc:  # pragma: no cover
            logger.error("pikepdf compression failed: %s", exc)
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            return None

        return self._finalize_compressed_pdf(pdf_path, tmp_path, min_ratio)

    def _compress_pdf_ghostscript(
        self,
        pdf_path: Path,
        method_config: Dict[str, Any],
        min_ratio: float,
    ) -> Optional[Path]:
        """Compress PDF via Ghostscript CLI."""
        gs_executable = method_config.get("path", "gs")
        if shutil.which(gs_executable) is None:
            logger.warning(
                "Ghostscript executable '%s' not found; skipping compression",
                gs_executable,
            )
            return None

        pdf_settings = method_config.get("pdf_settings", "/ebook")
        compatibility_level = str(method_config.get("compatibility_level", "1.4"))

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        cmd = [
            gs_executable,
            "-sDEVICE=pdfwrite",
            f"-dCompatibilityLevel={compatibility_level}",
            f"-dPDFSETTINGS={pdf_settings}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={tmp_path}",
            str(pdf_path),
        ]

        logger.info("Compressing PDF %s using Ghostscript", pdf_path.name)
        try:
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.error("Ghostscript compression failed: %s", exc)
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            return None

        return self._finalize_compressed_pdf(pdf_path, tmp_path, min_ratio)

    @staticmethod
    def _finalize_compressed_pdf(
        original_pdf: Path,
        compressed_pdf: Path,
        min_ratio: float,
    ) -> Optional[Path]:
        """Accept or discard a compressed PDF based on resulting size."""
        if not compressed_pdf.exists():
            logger.error("Compression pipeline did not produce an output file")
            return None

        original_size = original_pdf.stat().st_size
        compressed_size = compressed_pdf.stat().st_size

        if compressed_size >= original_size:
            logger.info(
                "Compression did not reduce size (original %d bytes, compressed %d bytes)",
                original_size,
                compressed_size,
            )
            try:
                compressed_pdf.unlink()
            except FileNotFoundError:
                pass
            return None

        ratio = compressed_size / original_size
        if ratio > min_ratio:
            logger.info(
                "Compression ratio %.4f exceeds threshold %.4f; discarding compressed PDF",
                ratio,
                min_ratio,
            )
            try:
                compressed_pdf.unlink()
            except FileNotFoundError:
                pass
            return None

        logger.info(
            "Compression succeeded (original: %d bytes, compressed: %d bytes, ratio %.4f)",
            original_size,
            compressed_size,
            ratio,
        )
        return compressed_pdf

    def _encode_pdf_to_data_url(self, pdf_path: Path) -> str:
        """Encode a PDF to a base64 data URL suitable for OpenRouter chat API."""
        with pdf_path.open("rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:application/pdf;base64,{encoded}"

    def _call_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        pdf_filename: str,
        pdf_data_url: str,
        plugins: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request with embedded PDF."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.default_headers)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "file",
                        "file": {"filename": pdf_filename, "file_data": pdf_data_url},
                    },
                ],
            },
        ]

        payload: Dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config.get("temperature", 0.3),
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if plugins:
            payload["plugins"] = plugins

        response = requests.post(
            self.chat_completions_url,
            headers=headers,
            json=payload,
            timeout=float(self.config.get("request_timeout", 180)),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Chat completion failed ({response.status_code}): {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text[:2000]
            raise RuntimeError(
                f"Failed to parse OpenRouter response as JSON: {exc}. Raw response snippet: {snippet}"
            ) from exc

    @staticmethod
    def _extract_chat_content(response_json: Dict[str, Any]) -> str:
        """Normalize chat completion response content into a single string."""
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            content = content.strip()
            if content:
                return content

        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
            joined = "\n".join(texts).strip()
            if joined:
                return joined

        # Some providers return reasoning text even when content is blank.
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str):
            reasoning = reasoning.strip()
            if reasoning:
                return reasoning
        elif isinstance(reasoning, list):
            reasoning_text = "\n".join(str(item) for item in reasoning).strip()
            if reasoning_text:
                return reasoning_text

        return ""

    @staticmethod
    def _extract_finish_reason(response_json: Dict[str, Any]) -> Optional[str]:
        """Extract finish reason from chat completion response, if available."""
        choices = response_json.get("choices", [])
        if not choices:
            return None
        return choices[0].get("finish_reason")

    def _fallback_generate(
        self, pdf_path: str, metadata: Dict[str, Any], output_dir: str
    ) -> str:
        """Fallback to text-based extraction when PDF upload is not feasible."""
        if self._fallback_agent is None:
            from aust.src.agents.paper_card_agent import PaperCardAgent

            logger.info(
                "Initializing fallback text extraction agent with config %s",
                self.fallback_config_path,
            )
            self._fallback_agent = PaperCardAgent(
                config_path=self.fallback_config_path,
                model=self.config.get("model"),
                temperature=self.config.get("temperature", 0.3),
            )

        return self._fallback_agent.generate_card(
            pdf_path=pdf_path,
            metadata=metadata,
            output_dir=output_dir,
        )
