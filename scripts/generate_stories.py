#!/usr/bin/env python3
"""
Generate user story files from epic markdown files.

This script parses epic files and creates individual story markdown files
with proper structure, dev notes from architecture, and placeholders for
development tracking.
"""

import re
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class StoryGenerator:
    def __init__(self, epics_dir: str, stories_dir: str, architecture_dir: str):
        self.epics_dir = Path(epics_dir)
        self.stories_dir = Path(stories_dir)
        self.architecture_dir = Path(architecture_dir)
        self.stories_dir.mkdir(parents=True, exist_ok=True)

    def parse_epic_file(self, epic_path: Path) -> Dict:
        """Parse an epic file and extract stories."""
        with open(epic_path, 'r') as f:
            content = f.read()

        # Extract epic number and title from filename or content
        epic_match = re.search(r'# Epic (\d+): (.+)', content)
        if not epic_match:
            raise ValueError(f"Could not find epic title in {epic_path}")

        epic_num = epic_match.group(1)
        epic_title = epic_match.group(2).strip()

        # Extract epic goal
        goal_match = re.search(r'\*\*Expanded Goal\*\*: (.+?)(?=\n\n##)', content, re.DOTALL)
        epic_goal = goal_match.group(1).strip() if goal_match else ""

        # Split content by story sections (## Story X.X:)
        story_pattern = r'## Story (\d+\.\d+): (.+?)\n\n(.*?)(?=\n## Story |\n# |$)'
        stories = []

        for match in re.finditer(story_pattern, content, re.DOTALL):
            story_num = match.group(1)
            story_title = match.group(2).strip()
            story_content = match.group(3).strip()

            # Parse story components
            story_data = self.parse_story_content(story_num, story_title, story_content)
            stories.append(story_data)

        return {
            'epic_num': epic_num,
            'epic_title': epic_title,
            'epic_goal': epic_goal,
            'stories': stories
        }

    def parse_story_content(self, story_num: str, story_title: str, content: str) -> Dict:
        """Parse individual story content."""
        # Extract "As a... I want... so that..."
        as_match = re.search(r'As a \*\*(.+?)\*\*,', content)
        want_match = re.search(r'I want \*\*(.+?)\*\*,', content)
        so_match = re.search(r'so that \*\*(.+?)\*\*\.', content)

        role = as_match.group(1) if as_match else "user"
        action = want_match.group(1) if want_match else ""
        benefit = so_match.group(1) if so_match else ""

        # Extract acceptance criteria
        ac_section = re.search(r'### Acceptance Criteria\n\n(.*?)(?=\n## |\n### |$)', content, re.DOTALL)
        ac_text = ac_section.group(1).strip() if ac_section else ""

        # Parse numbered AC items
        ac_items = re.findall(r'^\d+\.\s+(.+?)$', ac_text, re.MULTILINE)

        return {
            'story_num': story_num,
            'story_title': story_title,
            'role': role,
            'action': action,
            'benefit': benefit,
            'acceptance_criteria': ac_items
        }

    def generate_story_file(self, epic_data: Dict, story_data: Dict):
        """Generate a story markdown file."""
        story_num = story_data['story_num']
        story_title = story_data['story_title']

        # Create filename
        title_slug = re.sub(r'[^\w\s-]', '', story_title.lower())
        title_slug = re.sub(r'[-\s]+', '-', title_slug)
        filename = f"{story_num}.{title_slug}.md"
        filepath = self.stories_dir / filename

        # Skip if already exists (don't overwrite manually created stories)
        if filepath.exists():
            print(f"  ⏭️  Skipping {filename} (already exists)")
            return

        # Generate content
        content = self.build_story_content(epic_data, story_data)

        # Write file
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✅ Created {filename}")

    def build_story_content(self, epic_data: Dict, story_data: Dict) -> str:
        """Build the complete story markdown content."""
        story_num = story_data['story_num']
        story_title = story_data['story_title']
        role = story_data['role']
        action = story_data['action']
        benefit = story_data['benefit']
        ac_items = story_data['acceptance_criteria']

        # Build acceptance criteria
        ac_text = "\n".join([f"{i+1}. {ac}" for i, ac in enumerate(ac_items)])

        # Generate tasks/subtasks
        tasks = self.generate_tasks(story_data)

        # Generate dev notes
        dev_notes = self.generate_dev_notes(story_num, story_title)

        # Build complete content
        content = f"""# Story {story_num}: {story_title}

## Status

Draft

## Story

**As a** {role},
**I want** {action},
**so that** {benefit}.

## Acceptance Criteria

{ac_text}

## Tasks / Subtasks

{tasks}

## Dev Notes

{dev_notes}

### Testing

From [architecture/test-strategy-and-standards.md](../architecture/test-strategy-and-standards.md):

- **Framework:** pytest 7.4.3
- **File Convention:** `tests/unit/test_{{module}}.py`
- **Location:** `tests/unit/` directory
- **Mocking Library:** pytest-mock 3.12.0

**Unit Test Requirements:**
- Follow AAA pattern (Arrange, Act, Assert)
- Mock all external dependencies
- Test file naming: `test_{{function_name}}_{{scenario}}_{{expected_result}}`
- Cover edge cases and error conditions

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| {datetime.now().strftime('%Y-%m-%d')} | 0.1 | Initial story creation | Sarah (PO Agent) |

## Dev Agent Record

*This section will be populated by the development agent during implementation*

### Agent Model Used

*TBD*

### Debug Log References

*TBD*

### Completion Notes List

*TBD*

### File List

*TBD*

## QA Results

*This section will be populated by QA Agent after implementation*
"""
        return content

    def generate_tasks(self, story_data: Dict) -> str:
        """Generate task checklist from acceptance criteria."""
        ac_items = story_data['acceptance_criteria']

        if not ac_items:
            return "- [ ] Implement story functionality\n  - [ ] Add implementation details as subtasks"

        tasks = []
        for i, ac in enumerate(ac_items, 1):
            # Create main task from AC
            task = f"- [ ] {ac} (AC: {i})"
            tasks.append(task)
            # Add placeholder subtask
            tasks.append(f"  - [ ] Add implementation subtasks")

        return "\n".join(tasks)

    def generate_dev_notes(self, story_num: str, story_title: str) -> str:
        """Generate dev notes section with architecture references."""
        notes = f"""### Relevant Architecture

This story should reference the following architecture documents:

- [Source Tree](../architecture/source-tree.md) - For directory structure and file locations
- [Components](../architecture/components.md) - For component specifications and interfaces
- [Data Models](../architecture/data-models.md) - For Pydantic model definitions
- [Tech Stack](../architecture/tech-stack.md) - For technology versions and dependencies
- [Coding Standards](../architecture/coding-standards.md) - For style and critical rules
- [Error Handling Strategy](../architecture/error-handling-strategy.md) - For exception handling patterns

### Key Implementation Notes

**Developer:** Review the architecture documents above and add specific implementation notes here:
- Relevant source tree paths for this story
- Specific components or data models to implement
- Technology stack requirements
- Critical coding standards to follow
- Error handling patterns to use

### Coding Standards Reminder

From [architecture/coding-standards.md](../architecture/coding-standards.md):

**Critical Rules:**
- Never use print() in production code - use logging module
- All Pydantic models must have validation
- Never hardcode API keys or secrets
- All datetime objects must be timezone-aware
- Type hints mandatory for all function signatures
"""
        return notes

    def process_all_epics(self):
        """Process all epic files and generate story files."""
        epic_files = sorted(self.epics_dir.glob('epic-*.md'))

        if not epic_files:
            print(f"❌ No epic files found in {self.epics_dir}")
            return

        print(f"\n📚 Processing {len(epic_files)} epic file(s)...\n")

        total_stories = 0
        for epic_file in epic_files:
            print(f"📖 Processing {epic_file.name}...")
            try:
                epic_data = self.parse_epic_file(epic_file)
                epic_num = epic_data['epic_num']
                epic_title = epic_data['epic_title']
                stories = epic_data['stories']

                print(f"   Epic {epic_num}: {epic_title} ({len(stories)} stories)")

                for story_data in stories:
                    self.generate_story_file(epic_data, story_data)
                    total_stories += 1

                print()
            except Exception as e:
                print(f"   ❌ Error processing {epic_file.name}: {e}")
                continue

        print(f"\n✨ Story generation complete! Total stories processed: {total_stories}")
        print(f"   Stories directory: {self.stories_dir}")


def main():
    """Main entry point."""
    # Paths relative to CAUST root
    epics_dir = "docs/epics"
    stories_dir = "docs/stories"
    architecture_dir = "docs/architecture"

    print("=" * 60)
    print("AUST Story Generator")
    print("=" * 60)

    generator = StoryGenerator(epics_dir, stories_dir, architecture_dir)
    generator.process_all_epics()


if __name__ == "__main__":
    main()
