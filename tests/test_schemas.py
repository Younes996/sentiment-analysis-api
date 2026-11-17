import unittest

from pydantic import ValidationError

from app.schemas import TextInput


class TestTextInput(unittest.TestCase):
    
    def test_rejects_whitespace_only_text(self):
        with self.assertRaises(ValidationError):
            TextInput(text="   ")
    
    def test_rejects_empty_text(self):
        with self.assertRaises(ValidationError):
            TextInput(text="")

    def test_accepts_text_with_surrounding_whitespace(self):
        text = "  Great movie!  "
        payload = TextInput(text=text)

        self.assertEqual(payload.text, text)