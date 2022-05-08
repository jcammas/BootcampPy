from TinyStatistician import TinyStatistician


data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
epsilon = 1e-5
err = "Error, grade 0 :("
tstat = TinyStatistician()
assert abs(tstat.mean(data) - 499.2) < epsilon, err

assert abs(tstat.median(data) - 210.5) < epsilon, err

quartile = tstat.quartile(data)
assert abs(quartile[0] - 18) < epsilon, err
assert abs(quartile[1] - 650) < epsilon, err
assert abs(tstat.percentile(data, 10) - 3) < epsilon, err
assert abs(tstat.percentile(data, 28) - 18) < epsilon, err
assert abs(tstat.percentile(data, 83) - 754) < epsilon, err
assert abs(tstat.var(data) - 589194.56) < epsilon, err
assert abs(tstat.std(data) - 767.5900989460456) < epsilon, err


data = [2, 37, 69, 318, 32, 3, 65, 74, 38, 26, 1, 79]
print(f"{len(data) = }")
print(f"{(len(data) / 2) % 2 == 0}")
quartile = tstat.quartile(data)
print(f"{quartile[0] = }")
print(f"{quartile[1] = }")

print(f"{tstat.percentile(data, 2) = }")
print(f"{tstat.percentile(data, 28) = }")
print(f"{tstat.percentile(data, 83) = }")
