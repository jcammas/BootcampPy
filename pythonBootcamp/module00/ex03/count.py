import sys
from string import punctuation


def text_analyzer(text="default", *args):
    """Tu dois des sous"""
    if args:
        print("ERROR")
        sys.exit(0)
    if text == "default":
        text = input("What is the text to analyze?\n")
    try:
        x = len(text)
        uppers = len([letter for letter in text if letter.isupper()])
        lowers = len([letter for letter in text if letter.islower()])
        puncs = len([letter for letter in text if letter in punctuation])
        spaces = len([letter for letter in text if letter == " "])
        print(f"""\
The text contains {x} characters:
    - {uppers} upper letters
    - {lowers} lower letters
    - {puncs} punctuation marks
    - {spaces} spaces""")
    except TypeError:
        print("ERROR")
