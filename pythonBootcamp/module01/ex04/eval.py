class Evaluator:
    """doc"""

    @staticmethod
    def zip_evaluate(coefs, words):
        if len(coefs) != len(words):
            return -1
        return sum([len(word) * coef for coef, word in zip(coefs, words)])

    @staticmethod
    def enumerate_evaluate(coefs, words):
        if coefs == None and words == None:
            return -1
        for x in words:
            if isinstance(x, int):
                return -1
        if len(coefs) != len(words):
            return -1
        res = 0
        for i, word in enumerate(words):
            res += len(word) * coefs[i]
        return res


print(Evaluator.enumerate_evaluate(None, None))
print(Evaluator.enumerate_evaluate([1, 2, 3], []))
print(Evaluator.enumerate_evaluate([1, 2, 3], ["word", 2, "wordo"]))

words = ["Le", "Lorem", "Ipsum", "est", "simple"]
coefs = [1.0, 2.0, 1.0, 4.0, 0.5]
print(Evaluator.zip_evaluate(coefs, words))

words = ["Le", "Lorem", "Ipsum", "n’", "est", "pas", "simple"]
coefs = [0.0, -1.0, 1.0, -12.0, 0.0, 42.42]
print(Evaluator.enumerate_evaluate(coefs, words))
