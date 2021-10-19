import sys
import string

MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                   'C': '-.-.', 'D': '-..', 'E': '.',
                   'F': '..-.', 'G': '--.', 'H': '....',
                   'I': '..', 'J': '.---', 'K': '-.-',
                   'L': '.-..', 'M': '--', 'N': '-.',
                   'O': '---', 'P': '.--.', 'Q': '--.-',
                   'R': '.-.', 'S': '...', 'T': '-',
                   'U': '..-', 'V': '...-', 'W': '.--',
                   'X': '-..-', 'Y': '-.--', 'Z': '--..',
                   '1': '.----', '2': '..---', '3': '...--',
                   '4': '....-', '5': '.....', '6': '-....',
                   '7': '--...', '8': '---..', '9': '----.',
                   '0': '-----'}
cipher = ""
for message in sys.argv[1:]:
    for letter in message:
        if letter in MORSE_CODE_DICT:
            cipher += MORSE_CODE_DICT[letter]
        elif letter.upper() in MORSE_CODE_DICT:
            cipher += MORSE_CODE_DICT[letter.upper()]
        elif letter is ' ':
            cipher += "/"
        else:
            sys.exit("ERROR")
        cipher += " "
    cipher += "/"
    cipher += " "
cipher = cipher[:-2]
print(cipher)
