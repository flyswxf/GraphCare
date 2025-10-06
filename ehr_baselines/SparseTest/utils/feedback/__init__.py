"""
Feedback processing utilities package.

Contains modules to:
- Read user natural-language feedback from response file.
- Extract two keyword groups (add/remove) with controllable counts.
- Map keywords to most related cluster indices using embeddings.

All file paths are centralized in constants.py for easy reconfiguration.
"""