from sys import argv as av


ag = [string.upper() for string in av[1:]]

MORSE_CODE_DICT = {'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
                        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
                        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
                        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
                        'Y': '-.--', 'Z': '--..', ' ': '/',
                        '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
                        '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----'}
tab = []

try:
    for string in ag:
        tab.append(" ".join([MORSE_CODE_DICT[letter] for letter in string]))
except KeyError:
    print("ERROR")
    exit()

tab = " / ".join(tab)

if len(av) <= 1:
    exit()

print(tab)
