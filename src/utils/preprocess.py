from .constants import TONENORMALIZE_DICT
import re
import unicodedata




def normalize_text_to_nfc_standard(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def normalize_text_to_nfkc_standard(text: str) -> str:
    return unicodedata.normalize('NFKC', text)


def normalize_tone(text: str) -> str:
    for tone, tone_replace in TONENORMALIZE_DICT:
        text = text.replace(tone, tone_replace)
    return text


def preprocess(question: str, lowercase: bool=False):
    if lowercase:
        question = question.lower()
    question = normalize_text_to_nfc_standard(question)
    question = normalize_tone(question)
    return question


if __name__ == '__main__':
    test_string = """
    
    """
    with open("output.txt", "w") as text_file:
        print(preprocess(test_string), file=text_file)
