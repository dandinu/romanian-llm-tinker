#!/usr/bin/env python3
"""
Unit tests for data processing functions in prepare_data.py.

Tests cover:
- Text cleaning and normalization
- Romanian language detection
- Text validation
- Q&A pair extraction
- JSONL validation
"""

import json
import pytest
import tempfile
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from prepare_data import RomanianDataProcessor, DataConstants


class TestTextCleaning:
    """Test text cleaning and normalization functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_clean_text_removes_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        text = "Text   with    multiple     spaces"
        result = self.processor.clean_text(text)
        assert "   " not in result
        assert result == "Text with multiple spaces"

    def test_clean_text_preserves_romanian_diacritics(self):
        """Test that Romanian diacritics are preserved."""
        text = "Aceasta este o propoziție cu diacritice: ăâîșț ĂÂÎȘȚ"
        result = self.processor.clean_text(text)
        # Check all Romanian diacritics are present
        assert 'ă' in result
        assert 'â' in result
        assert 'î' in result
        assert 'ș' in result
        assert 'ț' in result

    def test_clean_text_normalizes_quotes(self):
        """Test that various quote styles are normalized."""
        text = '„text with different quotes" «more quotes»'
        result = self.processor.clean_text(text)
        # Should convert to standard quotes
        assert '„' not in result
        assert '«' not in result
        assert '»' not in result

    def test_clean_text_fixes_punctuation_spacing(self):
        """Test that punctuation spacing is corrected."""
        text = "Text with bad spacing ,and !punctuation ."
        result = self.processor.clean_text(text)
        # Punctuation should be attached to word before, space after
        assert " ," not in result
        assert " !" not in result
        assert " ." not in result

    def test_clean_text_strips_leading_trailing(self):
        """Test that leading/trailing whitespace is removed."""
        text = "   text with spaces   "
        result = self.processor.clean_text(text)
        assert result == "text with spaces"

    def test_clean_text_empty_string(self):
        """Test cleaning empty string."""
        result = self.processor.clean_text("")
        assert result == ""


class TestRomanianValidation:
    """Test Romanian language detection and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_is_valid_text_accepts_good_romanian(self):
        """Test that valid Romanian text is accepted."""
        text = "Aceasta este o propoziție validă în limba română cu diacritice."
        assert self.processor.is_valid_text(text) is True

    def test_is_valid_text_rejects_too_short(self):
        """Test that text below minimum length is rejected."""
        text = "abc"  # Less than default min_length=10
        assert self.processor.is_valid_text(text) is False

    def test_is_valid_text_rejects_too_long(self):
        """Test that text above maximum length is rejected."""
        text = "a" * 5000  # More than default max_length=4096
        assert self.processor.is_valid_text(text) is False

    def test_is_valid_text_rejects_no_diacritics(self):
        """Test that text without Romanian diacritics is rejected."""
        text = "This is English text without Romanian diacritics."
        assert self.processor.is_valid_text(text) is False

    def test_is_valid_text_rejects_no_punctuation(self):
        """Test that text without sentence punctuation is rejected."""
        text = "Text fără punctuație finală de propoziție"
        assert self.processor.is_valid_text(text) is False

    def test_is_valid_text_accepts_various_punctuation(self):
        """Test that text with various punctuation marks is accepted."""
        text1 = "Propoziție cu punct final și diacritice."
        text2 = "Întrebare cu semn de întrebare?"
        text3 = "Exclamație cu semn de exclamare!"

        assert self.processor.is_valid_text(text1) is True
        assert self.processor.is_valid_text(text2) is True
        assert self.processor.is_valid_text(text3) is True


class TestQAPairExtraction:
    """Test Q&A pair extraction from Wikipedia text."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_extract_qa_pairs_from_wiki_basic(self):
        """Test basic Q&A pair extraction."""
        text = """România este o țară în Europa de Est.

        Are o populație de aproximativ 19 milioane de locuitori.

        Capitala României este București."""

        title = "România"

        pairs = self.processor.extract_qa_pairs_from_wiki(text, title)

        # Should generate at least some pairs
        assert len(pairs) > 0

        # Each pair should have instruction, response, and topic
        for pair in pairs:
            assert 'instruction' in pair
            assert 'response' in pair
            assert 'topic' in pair
            assert pair['topic'] == title

    def test_extract_qa_pairs_includes_topic_in_instruction(self):
        """Test that topic is included in instruction."""
        text = "Text valid în limba română cu diacritice. " * 20
        title = "Test Topic"

        pairs = self.processor.extract_qa_pairs_from_wiki(text, title)

        # At least one instruction should contain the topic
        assert any(title.lower() in pair['instruction'].lower() for pair in pairs)

    def test_extract_qa_pairs_truncates_long_responses(self):
        """Test that long responses are truncated."""
        # Create very long text
        long_paragraph = "Text foarte lung în limba română cu diacritice. " * 100
        text = long_paragraph
        title = "Long Text"

        pairs = self.processor.extract_qa_pairs_from_wiki(text, title)

        # Responses should be truncated
        for pair in pairs:
            assert len(pair['response']) <= DataConstants.MAX_RESPONSE_LENGTH + 3  # +3 for "..."

    def test_extract_qa_pairs_rejects_invalid_text(self):
        """Test that invalid text returns no pairs."""
        text = "Invalid text"  # Too short, no diacritics, wrong language
        title = "Invalid"

        pairs = self.processor.extract_qa_pairs_from_wiki(text, title)

        assert len(pairs) == 0

    def test_extract_qa_pairs_filters_short_paragraphs(self):
        """Test that short paragraphs are filtered out."""
        text = "Short.\nText în română cu diacritice. " * 20
        title = "Test"

        pairs = self.processor.extract_qa_pairs_from_wiki(text, title)

        # Should have extracted pairs from valid paragraphs
        # Short paragraphs should be filtered
        for pair in pairs:
            assert len(pair['response']) > DataConstants.MIN_PARAGRAPH_LENGTH


class TestInstructionExampleCreation:
    """Test instruction example creation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_create_instruction_example_basic(self):
        """Test basic instruction example creation."""
        instruction = "Explică ce este AI."
        response = "AI este inteligența artificială."

        example = self.processor.create_instruction_example(instruction, response)

        assert 'messages' in example
        assert len(example['messages']) == 2
        assert example['messages'][0]['role'] == 'user'
        assert example['messages'][0]['content'] == instruction
        assert example['messages'][1]['role'] == 'assistant'
        assert example['messages'][1]['content'] == response

    def test_create_instruction_example_with_system_prompt(self):
        """Test instruction example with system prompt."""
        instruction = "Explică AI."
        response = "AI este inteligența artificială."
        system_prompt = "Tu ești un asistent util."

        example = self.processor.create_instruction_example(
            instruction, response, system_prompt
        )

        assert len(example['messages']) == 3
        assert example['messages'][0]['role'] == 'system'
        assert example['messages'][0]['content'] == system_prompt


class TestJSONLValidation:
    """Test JSONL file validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_validate_jsonl_valid_file(self):
        """Test validation of a valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write valid examples
            examples = [
                {
                    'messages': [
                        {'role': 'user', 'content': 'Question?'},
                        {'role': 'assistant', 'content': 'Answer.'}
                    ]
                },
                {
                    'messages': [
                        {'role': 'system', 'content': 'System prompt'},
                        {'role': 'user', 'content': 'Question?'},
                        {'role': 'assistant', 'content': 'Answer.'}
                    ]
                }
            ]

            for ex in examples:
                f.write(json.dumps(ex) + '\n')

            temp_path = Path(f.name)

        try:
            stats = self.processor.validate_jsonl(temp_path)

            assert stats['total_lines'] == 2
            assert stats['valid_lines'] == 2
            assert stats['invalid_lines'] == 0
            assert len(stats['errors']) == 0
        finally:
            temp_path.unlink()

    def test_validate_jsonl_missing_messages_field(self):
        """Test validation catches missing 'messages' field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write invalid example - missing 'messages'
            f.write(json.dumps({'invalid': 'data'}) + '\n')
            temp_path = Path(f.name)

        try:
            stats = self.processor.validate_jsonl(temp_path)

            assert stats['total_lines'] == 1
            assert stats['valid_lines'] == 0
            assert stats['invalid_lines'] == 1
            assert len(stats['errors']) > 0
        finally:
            temp_path.unlink()

    def test_validate_jsonl_invalid_role(self):
        """Test validation catches invalid role."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write example with invalid role
            example = {
                'messages': [
                    {'role': 'invalid_role', 'content': 'Text'}
                ]
            }
            f.write(json.dumps(example) + '\n')
            temp_path = Path(f.name)

        try:
            stats = self.processor.validate_jsonl(temp_path)

            assert stats['invalid_lines'] == 1
            assert len(stats['errors']) > 0
        finally:
            temp_path.unlink()

    def test_validate_jsonl_missing_content(self):
        """Test validation catches missing content field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write example with missing content
            example = {
                'messages': [
                    {'role': 'user'}  # Missing 'content'
                ]
            }
            f.write(json.dumps(example) + '\n')
            temp_path = Path(f.name)

        try:
            stats = self.processor.validate_jsonl(temp_path)

            assert stats['invalid_lines'] == 1
        finally:
            temp_path.unlink()

    def test_validate_jsonl_malformed_json(self):
        """Test validation handles malformed JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write malformed JSON
            f.write('{"invalid": json}\n')
            temp_path = Path(f.name)

        try:
            stats = self.processor.validate_jsonl(temp_path)

            assert stats['invalid_lines'] == 1
        finally:
            temp_path.unlink()


class TestTrainValSplit:
    """Test train/validation split functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RomanianDataProcessor()

    def test_create_train_val_split_ratio(self):
        """Test that split ratio is correct."""
        examples = [{'data': i} for i in range(100)]

        train, val = self.processor.create_train_val_split(examples, split_ratio=0.8)

        assert len(train) == 80
        assert len(val) == 20

    def test_create_train_val_split_no_overlap(self):
        """Test that train and val sets don't overlap."""
        examples = [{'data': i} for i in range(100)]

        train, val = self.processor.create_train_val_split(examples, split_ratio=0.7)

        # Convert to sets of JSON strings for comparison
        train_set = {json.dumps(ex, sort_keys=True) for ex in train}
        val_set = {json.dumps(ex, sort_keys=True) for ex in val}

        # No overlap
        assert len(train_set & val_set) == 0

        # Combined should equal original
        assert len(train) + len(val) == len(examples)

    def test_create_train_val_split_seed_reproducible(self):
        """Test that same seed produces same split."""
        examples = [{'data': i} for i in range(100)]

        train1, val1 = self.processor.create_train_val_split(examples, seed=42)
        train2, val2 = self.processor.create_train_val_split(examples, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_create_train_val_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        examples = [{'data': i} for i in range(100)]

        train1, _ = self.processor.create_train_val_split(examples, seed=42)
        train2, _ = self.processor.create_train_val_split(examples, seed=123)

        # Should be different (very unlikely to be identical by chance)
        assert train1 != train2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
