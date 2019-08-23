import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

benchmarks = [
    ('standard', '../logs/cifar-standard-51200.txt'),
    ('smart 1%', '../logs/cifar-smart-1%-51200.txt'),
    ('exponential', '../logs/cifar-exponential-51200.txt')
]

def parse_file(filename):
    accuracies = []
    for line in open(filename, 'r').readlines():
        if "Accuracy:" not in line:
            continue
        accuracy = line.split(" ")[-2].split("/")
        accuracies.append(int(accuracy[0]) / int(accuracy[1]))
    return accuracies

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)

for benchmark in benchmarks:
    accuracies = parse_file(benchmark[1])
    epochs = range(1, len(accuracies) + 1)

    ax.plot(epochs, accuracies, label=benchmark[0])
    plt.legend()
    plt.savefig("accuracy-plot.png")

