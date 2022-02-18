from generator import generator

text = "Lorem Ipsum Lorem Ipsum"
for word in generator(text, sep=" ", option="unique"):
    print(word)
