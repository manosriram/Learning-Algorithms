import statistics as st


def normalize(a):
    mn = 0
    for i in a:
        mn += i

    res = mn / len(a)
    res = (mn - res) / st.stdev(a)
    return res


a = [1, 9381, 1231, 22, 232, 12312, 1823918239182]

print(normalize(a))
