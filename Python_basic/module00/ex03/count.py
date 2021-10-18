from string import punctuation


def text_analyzer(text="", *args):
    """Ceci est une doc."""
    if args:
        print("ERROR")
        return
    if text == "":
        text = input("What is the text to analyse?\n")
    try:
        chars = len(text)
        uppers = len([letter for letter in text if letter.isupper()])
        lowers = len([letter for letter in text if letter.islower()])
        puncs = len([letter for letter in text if letter in punctuation])
        spaces = len([letter for letter in text if letter == " "])
        print(f"""The text contains {chars} characters:
  - {uppers} upper letters
  - {lowers} lower letters
  - {puncs} punctuation marks
  - {spaces} spaces""")
    except:
        print("ERROR")
