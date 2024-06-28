import string
from .constants import *
import unicodedata


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def normalize_text_to_nfc_standard(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def normalize_text_to_nfkc_standard(text: str) -> str:
    return unicodedata.normalize('NFKC', text)


def normalize_tone(text: str) -> str:
    for tone, tone_replace in TONENORMALIZE_DICT:
        text = text.replace(tone, tone_replace)
    return text


def cleanhtml(raw_page):
    out = CLOSE_TAG.sub("", raw_page)
    out = OPEN_TAG.sub("", out)
    return out


def preprocess(
    text: str, 
    lowercase: bool=False,
    remove_punct: bool=False
):
    if lowercase:
        text = text.lower()
    text = normalize_text_to_nfc_standard(text)
    text = normalize_tone(text)
    if remove_punct:
        text = remove_punctuation(text)
    return text


if __name__ == '__main__':
    test_string = """
    
    """
    with open("output.txt", "w") as text_file:
        print(preprocess(test_string), file=text_file)
