from PIL import Image, ImageFilter
import random

#P no of colors in Pallete 8,12,24 etc
#N no of cells or detail level
def draw(file_name, P, N, M=3):
    img = Image.open(file_name, 'r')
    pixels = img.load()
    size_x, size_y = img.size

    def dist(c1, c2):
        return (c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2

    def mean(colours):
        n = len(colours)
        r = sum(c[0] for c in colours)//n
        g = sum(c[1] for c in colours)//n
        b = sum(c[2] for c in colours)//n
        return (r, g, b)

    def colourize(colour, palette):
        return min(palette, key=lambda c: dist(c, colour))

    def cluster(colours, k, max_n=10000, max_i=10):
        colours = random.sample(colours, max_n)
        centroids = random.sample(colours, k)
        i = 0
        old_centroids = None
        while not(i > max_i or centroids == old_centroids):
            old_centroids = centroids
            i += 1
            labels = [colourize(c, centroids) for c in colours]
            centroids = [mean([c for c, l in zip(colours, labels)
                               if l is cen]) for cen in centroids]
        return centroids

    all_coords = [(x, y) for x in range(size_x) for y in range(size_y)]
    all_colours = [pixels[x, y] for x, y in all_coords]
    palette = cluster(all_colours, P)
    print ('clustered')

    for x, y in all_coords:
        pixels[x, y] = colourize(pixels[x, y], palette)
    print ('colourized')

    median_filter = ImageFilter.MedianFilter(size=M)
    img = img.filter(median_filter)
    pixels = img.load()
    for x, y in all_coords:
        pixels[x, y] = colourize(pixels[x, y], palette)
    print ('median filtered')

    def neighbours(edge, outer, colour=None):
        return set((x+a, y+b) for x, y in edge
                   for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if (x+a, y+b) in outer
                   and (colour == None or pixels[(x+a, y+b)] == colour))

    def cell(centre, rest):
        colour = pixels[centre]
        edge = set([centre])
        region = set()
        while edge:
            region |= edge
            rest = rest-edge
            edge = set(n for n in neighbours(edge, rest, colour))
        return region, rest

    print ('start segmentation:')
    rest = set(all_coords)
    cells = []
    while rest:
        centre = random.sample(rest, 1)[0]
        region, rest = cell(centre, rest-set(centre))
        cells += [region]
        print ('%d pixels remaining' % len(rest))
    cells = sorted(cells, key=len, reverse=True)
    print ('segmented (%d segments)' % len(cells))

    print ('start merging:')
    while len(cells) > N:
        small_cell = cells.pop()
        n = neighbours(small_cell, set(all_coords)-small_cell)
        for big_cell in cells:
            if big_cell & n:
                big_cell |= small_cell
                break
        print ('%d segments remaining' % len(cells))
    print ('merged')

    for cell in cells:
        colour = colourize(mean([pixels[x, y] for x, y in cell]), palette)
        for x, y in cell:
            pixels[x, y] = colour
    print ('colorized again')

    img.save('P%d N%d ' % (P, N)+file_name)
    print ('saved')


draw('./leafs.JPG', 8, 500, 1)
