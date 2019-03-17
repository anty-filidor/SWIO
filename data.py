import csv
import numpy as np
import matplotlib.pyplot as plt


def import_file(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(1, len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


data = import_file('dane.csv')


"""RAW DATA"""
#print("data: ", np.shape(data), type(data), data)
area_1 = []
aspect_1 = []
for i in range(1, len(data)):
    aa, at = data[i]
    area_1.append(aa)
    aspect_1.append(at)

#print("area: ", np.shape(area), type(area), area)
#print("aspect: ", np.shape(aspect), type(aspect), aspect)

"""EXCLUDE_BOUNDARIES_OF_IMAGE"""
area_2 = []
aspect_2 = []
for i in range(1, len(data)):
    aa, at = data[i]
    if aa < 4000:
        area_2.append(aa)
        aspect_2.append(at)


plt.figure(1)

plt.title('Aspect by Area')
plt.xlabel('Area')
plt.ylabel('Aspect ratio')
plt.plot(area_2, aspect_2, 'o')
plt.show()

# plotowanie wykresów
