"""
Utility for parsing structured markdown documents and extracting specific sections.

This module provides functions to extract sections from markdown-formatted paper cards
based on heading patterns, enabling selective inclusion of relevant content.
"""

import re
from typing import Any


def extract_markdown_sections(
    markdown_text: str,
    sections_to_extract: list[str],
    *,
    case_sensitive: bool = False,
) -> str:
    """
    Extract specific sections from a markdown document.

    Args:
        markdown_text: The full markdown content.
        sections_to_extract: List of section names to extract (e.g., ["METADATA", "SUMMARY"]).
        case_sensitive: Whether to match section names case-sensitively.

    Returns:
        A string containing only the requested sections, preserving markdown formatting.

    Examples:
        >>> text = "## [METADATA]\\nFoo\\n## [SUMMARY]\\nBar\\n## [DETAILS]\\nBaz"
        >>> extract_markdown_sections(text, ["METADATA", "SUMMARY"])
        '## [METADATA]\\nFoo\\n## [SUMMARY]\\nBar\\n'
    """
    if not markdown_text or not sections_to_extract:
        return ""

    # Build pattern to match section headers
    # Matches patterns like: ## [METADATA], ## [SUMMARY], etc.
    flags = 0 if case_sensitive else re.IGNORECASE

    # Create regex pattern to find all sections
    section_pattern = r'^(#{1,6})\s*\[([^\]]+)\]'

    lines = markdown_text.split('\n')
    extracted_lines = []
    current_section = None
    should_include = False
    section_indent_level = None

    for line in lines:
        # Check if this line is a section header
        match = re.match(section_pattern, line, flags)

        if match:
            header_level = len(match.group(1))  # Number of # symbols
            section_name = match.group(2).strip()

            # Check if this section should be extracted
            if case_sensitive:
                should_include = section_name in sections_to_extract
            else:
                should_include = any(
                    section_name.upper() == s.upper() for s in sections_to_extract
                )

            if should_include:
                current_section = section_name
                section_indent_level = header_level
                extracted_lines.append(line)
            else:
                # Check if we hit a new section at the same or higher level (fewer #)
                if section_indent_level is not None and header_level <= section_indent_level:
                    # Stop including lines from previous section
                    current_section = None
                    section_indent_level = None
                    should_include = False
        elif should_include and current_section is not None:
            # Include lines that are part of the current section
            extracted_lines.append(line)

    result = '\n'.join(extracted_lines)

    # Clean up extra blank lines at start/end
    return result.strip()


def extract_paper_card_sections(
    paper_dict: dict[str, Any],
    sections_to_extract: list[str],
) -> str:
    """
    Extract specific sections from a paper card dictionary.

    This function looks for markdown content in common fields like 'summary',
    'content', 'quick_summary', or the full card text, then extracts only
    the requested sections.

    Args:
        paper_dict: Dictionary containing paper card information.
        sections_to_extract: List of section names to extract (e.g.,
            ["METADATA", "SUMMARY", "CORE_METHOD"]).

    Returns:
        Formatted string with only the requested sections, or empty string
        if no markdown content is found.
    """
    # Try to find the full markdown content in various possible fields
    # 'summary' is the primary field used by query_generator._flatten_results
    markdown_content = None

    for field in ['summary', 'quick_summary', 'content', 'full_card', 'card_content', 'markdown', 'text']:
        if field in paper_dict and isinstance(paper_dict[field], str):
            markdown_content = paper_dict[field]
            break

    if not markdown_content:
        return ""

    return extract_markdown_sections(
        markdown_content,
        sections_to_extract,
        case_sensitive=False,
    )


__all__ = [
    'extract_markdown_sections',
    'extract_paper_card_sections',
]
