"""Trying Matplot Lib"""

from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

"""Create a line chart, years on x-axis, gdp on y-axis"""
plt.plot(years, gdp, color="purple", marker="o", linestyle="solid")

plt.title("Leidos Share Price")

plt.ylabel("Billions of $")
plt.show()


stocks = ["Apple", "Booz Allen", "AT&T", "Leidos"]
numbers = [1, 2, 3, 4]
xs = [i + .5 for i, _ in enumerate(stocks)]
plt.bar(xs, numbers)
plt.ylabel("Numbers")
plt.title("Stonks")
plt.xticks([i + .5 for i, _ in enumerate(stocks)], stocks)
plt.show()
