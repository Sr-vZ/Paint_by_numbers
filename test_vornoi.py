from collections import defaultdict
import itertools
import random
import time

from PIL import Image
import numpy as np
from scipy.spatial import KDTree, Delaunay

INFILE = "leafs.jpg"
OUTFILE = "voronoi.txt"
N = 3000

DEBUG = False  # Outputs extra images to see what's happening
FEATURE_FILE = "features.png"
SAMPLE_FILE = "samples.png"
SAMPLE_POINTS = 20000
ITERATIONS = 10
CLOSE_COLOR_THRESHOLD = 8

"""
Color conversion functions
"""

start_time = time.time()

# http://www.easyrgb.com/?X=MATH


def rgb2xyz(rgb):
  r, g, b = rgb
  r /= 255
  g /= 255
  b /= 255

  r = ((r + 0.055)/1.055)**2.4 if r > 0.04045 else r/12.92
  g = ((g + 0.055)/1.055)**2.4 if g > 0.04045 else g/12.92
  b = ((b + 0.055)/1.055)**2.4 if b > 0.04045 else b/12.92

  r *= 100
  g *= 100
  b *= 100

  x = r*0.4124 + g*0.3576 + b*0.1805
  y = r*0.2126 + g*0.7152 + b*0.0722
  z = r*0.0193 + g*0.1192 + b*0.9505

  return (x, y, z)


def xyz2lab(xyz):
  x, y, z = xyz
  x /= 95.047
  y /= 100
  z /= 108.883

  x = x**(1/3) if x > 0.008856 else 7.787*x + 16/116
  y = y**(1/3) if y > 0.008856 else 7.787*y + 16/116
  z = z**(1/3) if z > 0.008856 else 7.787*z + 16/116

  L = 116*y - 16
  a = 500*(x - y)
  b = 200*(y - z)

  return (L, a, b)


def rgb2lab(rgb):
  return xyz2lab(rgb2xyz(rgb))


def lab2xyz(lab):
  L, a, b = lab
  y = (L + 16)/116
  x = a/500 + y
  z = y - b/200

  y = y**3 if y**3 > 0.008856 else (y - 16/116)/7.787
  x = x**3 if x**3 > 0.008856 else (x - 16/116)/7.787
  z = z**3 if z**3 > 0.008856 else (z - 16/116)/7.787

  x *= 95.047
  y *= 100
  z *= 108.883

  return (x, y, z)


def xyz2rgb(xyz):
  x, y, z = xyz
  x /= 100
  y /= 100
  z /= 100

  r = x * 3.2406 + y*-1.5372 + z*-0.4986
  g = x*-0.9689 + y * 1.8758 + z * 0.0415
  b = x * 0.0557 + y*-0.2040 + z * 1.0570

  r = 1.055 * (r**(1/2.4)) - 0.055 if r > 0.0031308 else 12.92*r
  g = 1.055 * (g**(1/2.4)) - 0.055 if g > 0.0031308 else 12.92*g
  b = 1.055 * (b**(1/2.4)) - 0.055 if b > 0.0031308 else 12.92*b

  r *= 255
  g *= 255
  b *= 255

  return (r, g, b)


def lab2rgb(lab):
  return xyz2rgb(lab2xyz(lab))


"""
Step 1: Read image and convert to CIELAB
"""

im = Image.open(INFILE)
im = im.convert("RGB")
width, height = prev_size = im.size

pixlab_map = {}

for x in range(width):
    for y in range(height):
        pixlab_map[(x, y)] = rgb2lab(im.getpixel((x, y)))

print("Step 1: Image read and converted")

"""
Step 2: Get feature points
"""


def euclidean(point1, point2):
    return sum((x-y)**2 for x, y in zip(point1, point2))**.5


def neighbours(pixel):
    x, y = pixel
    results = []

    for dx, dy in itertools.product([-1, 0, 1], repeat=2):
        neighbour = (pixel[0] + dx, pixel[1] + dy)

        if (neighbour != pixel and 0 <= neighbour[0] < width
                and 0 <= neighbour[1] < height):
            results.append(neighbour)

    return results


def mse(colors, base):
    return sum(euclidean(x, base)**2 for x in colors)/len(colors)


features = []

for x in range(width):
    for y in range(height):
        pixel = (x, y)
        col = pixlab_map[pixel]
        features.append((mse([pixlab_map[n] for n in neighbours(pixel)], col),
                         random.random(),
                         pixel))

features.sort()
features_copy = [x[2] for x in features]

if DEBUG:
    test_im = Image.new("RGB", im.size)

    for i in range(len(features)):
        pixel = features[i][1]
        test_im.putpixel(pixel, (int(255*i/len(features)),)*3)

    test_im.save(FEATURE_FILE)

print("Step 2a: Edge detection-ish complete")


def random_index(list_):
    r = random.expovariate(2)

    while r > 1:
         r = random.expovariate(2)

    return int((1 - r) * len(list_))


sample_points = set()

while features and len(sample_points) < SAMPLE_POINTS:
    index = random_index(features)
    point = features[index][2]
    sample_points.add(point)
    del features[index]

print("Step 2b: {} feature samples generated".format(len(sample_points)))

if DEBUG:
    test_im = Image.new("RGB", im.size)

    for pixel in sample_points:
        test_im.putpixel(pixel, (255, 255, 255))

    test_im.save(SAMPLE_FILE)

"""
Step 3: Fuzzy k-means
"""


def euclidean(point1, point2):
    return sum((x-y)**2 for x, y in zip(point1, point2))**.5


def get_centroid(points):
    return tuple(sum(coord)/len(points) for coord in zip(*points))


def mean_cell_color(cell):
    return get_centroid([pixlab_map[pixel] for pixel in cell])


def median_cell_color(cell):
    # Pick start point out of mean and up to 10 pixels in cell
    mean_col = get_centroid([pixlab_map[pixel] for pixel in cell])
    start_choices = [pixlab_map[pixel] for pixel in cell]

    if len(start_choices) > 10:
        start_choices = random.sample(start_choices, 10)

    start_choices.append(mean_col)

    best_dist = None
    col = None

    for c in start_choices:
        dist = sum(euclidean(c, pixlab_map[pixel])
                   for pixel in cell)

        if col is None or dist < best_dist:
            col = c
            best_dist = dist

    # Approximate median by hill climbing
    last = None

    while last is None or euclidean(col, last) < 1e-6:
        last = col

        best_dist = None
        best_col = None

        for deviation in itertools.product([-1, 0, 1], repeat=3):
            new_col = tuple(x+y for x, y in zip(col, deviation))
            dist = sum(euclidean(new_col, pixlab_map[pixel])
                       for pixel in cell)

            if best_dist is None or dist < best_dist:
                best_col = new_col

        col = best_col

    return col


def random_point():
    index = random_index(features_copy)
    point = features_copy[index]

    dx = random.random() * 10 - 5
    dy = random.random() * 10 - 5

    return (point[0] + dx, point[1] + dy)


centroids = np.asarray([random_point() for _ in range(N)])
variance = {i: float("inf") for i in range(N)}
cluster_colors = {i: (0, 0, 0) for i in range(N)}

# Initial iteration
tree = KDTree(centroids)
clusters = defaultdict(set)

for point in sample_points:
    nearest = tree.query(point)[1]
    clusters[nearest].add(point)

# Cluster!
for iter_num in range(ITERATIONS):
    if DEBUG:
        test_im = Image.new("RGB", im.size)

        for n, pixels in clusters.items():
            color = 0xFFFFFF * (n/N)
            color = (int(color//256//256 % 256),
                     int(color//256 % 256), int(color % 256))

            for p in pixels:
                test_im.putpixel(p, color)

        test_im.save(SAMPLE_FILE)

    for cluster_num in clusters:
        if clusters[cluster_num]:
            cols = [pixlab_map[x] for x in clusters[cluster_num]]

            cluster_colors[cluster_num] = mean_cell_color(
                clusters[cluster_num])
            variance[cluster_num] = mse(cols, cluster_colors[cluster_num])

        else:
            cluster_colors[cluster_num] = (0, 0, 0)
            variance[cluster_num] = float("inf")

    print("Clustering (iteration {})".format(iter_num))

    # Remove useless/high variance
    if iter_num < ITERATIONS - 1:
        delaunay = Delaunay(np.asarray(centroids))
        neighbours = defaultdict(set)

        for simplex in delaunay.simplices:
            n1, n2, n3 = simplex

            neighbours[n1] |= {n2, n3}
            neighbours[n2] |= {n1, n3}
            neighbours[n3] |= {n1, n2}

        for num, centroid in enumerate(centroids):
            col = cluster_colors[num]

            like_neighbours = True

            nns = set()  # neighbours + neighbours of neighbours

            for n in neighbours[num]:
                nns |= {n} | neighbours[n] - {num}

            nn_far = sum(euclidean(col, cluster_colors[nn]) > CLOSE_COLOR_THRESHOLD
                         for nn in nns)

            if nns and nn_far / len(nns) < 1/5:
                sample_points -= clusters[num]

                for _ in clusters[num]:
                    if features and len(sample_points) < SAMPLE_POINTS:
                        index = random_index(features)
                        point = features[index][2]
                        sample_points.add(point)
                        del features[index]

                clusters[num] = set()

    new_centroids = []

    for i in range(N):
        if clusters[i]:
            new_centroids.append(get_centroid(clusters[i]))
        else:
            new_centroids.append(random_point())

    centroids = np.asarray(new_centroids)
    tree = KDTree(centroids)

    clusters = defaultdict(set)

    for point in sample_points:
        nearest = tree.query(point, k=6)[1]
        col = pixlab_map[point]

        for n in nearest:
            if n < N and euclidean(col, cluster_colors[n])**2 <= variance[n]:
                clusters[n].add(point)
                break

        else:
            clusters[nearest[0]].add(point)

print("Step 3: Fuzzy k-means complete")

"""
Step 4: Output
"""

for i in range(N):
    if clusters[i]:
        centroids[i] = get_centroid(clusters[i])

centroids = np.asarray(centroids)
tree = KDTree(centroids)
color_clusters = defaultdict(set)

# Throw back on some sample points to get the colors right
all_points = [(x, y) for x in range(width) for y in range(height)]

for pixel in random.sample(all_points, int(min(width*height, 5 * SAMPLE_POINTS))):
    nearest = tree.query(pixel)[1]
    color_clusters[nearest].add(pixel)

with open(OUTFILE, "w") as outfile:
    for i in range(N):
        if clusters[i]:
            centroid = tuple(centroids[i])
            col = tuple(
                x/255 for x in lab2rgb(median_cell_color(color_clusters[i] or clusters[i])))
            print(" ".join(map(str, centroid + col)), file=outfile)

print("Done! Time taken:", time.time() - start_time)
