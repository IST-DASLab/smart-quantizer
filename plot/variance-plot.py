import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statistics import mean

benchmarks = [
    ('standard', '../logs/variance-standard.txt'),
    ('exponential', '../logs/variance-exponential.txt'),
    ('smart 1%', '../logs/variance-smart-1%.txt'),
    ('smart 5%', '../logs/variance-smart-5%.txt'),
    ('smart 15%', '../logs/variance-smart-15%.txt')
]

def parse_file(filename):
    variancies = []
    for line in open(filename, 'r').readlines():
        if "Variance" not in line:
            continue
        variance = float(line.split(" ")[-1])
        variancies.append(variance)
    return variancies

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

for benchmark in benchmarks:
    variances = parse_file(benchmark[1])
    x = range(1, len(variances) + 1)

    ax.plot(x, variances, label=benchmark[0])
    print(benchmark[0], mean(variances))

plt.legend()
plt.savefig("variance-plot.png")
