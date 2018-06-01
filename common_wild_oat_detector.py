from math import sqrt
import os
import multiprocessing
import heapq
import timeit

import cv2
import numpy as np
import imutils

TARGET_DIR = '/home/makinen/utu/gradu/src/erikoistyo'
"""Directory where images with rois are written"""

def _get_connected_neighbours(y, x, width, height):
    """Returns the 8-connected neighbourhood of the pixel at (y,x).

    Parameters
    ----------
    y: int
        y coordinate of the pixel
    x:  int
        x coordinate of the pixel
    width: int
        width of the image mask
    height: int
        height of the image mask

    Returns
    -------
    list of tuples
        the coordinates of the neighbours
    """

    connected = []
    # north
    if y >= 0:
        connected.append((y-1, x))
    # south
    if y < height-2:
        connected.append((y+1, x))
    # west
    if x > 0:
        connected.append((y, x-1))
    # east
    if x < width-2:
        connected.append((y, x+1))
    # north east
    if y > 0 and x < width-2:
        connected.append((y-1, x+1))
    # south east
    if y < height-2 and x < width-2:
        connected.append((y+1, x+1))
    # north west
    if y > 0 and x > 0:
        connected.append((y-1, x-1))
    # north east
    if y < height -2 and x < width-2:
        connected.append((y+1, x+1))

    return connected


def _resize_seed_mask(seed, seed_mask):
    """Creates a new smaller seed mask and draws the given seed on it. This is
    needed because huge seed masks kill the performance of region growing.

    Parameters
    ----------
    seed: list
        List returned by findContours. Contains all boundary points as separate lists.
    seed_mask: numpy array
        original seed mask

    Returns
    -------
    (numpy array, func)
        Resized seed mask and a converter function which can be used to map
        the coordinates of the original mask to the smaller mask.
    """

    # creates a list of the pixel coordinates
    coordinates = []
    (rx, ry, rw, rh) = cv2.boundingRect(seed)

    for y in range(ry, ry+rh):
        for x in range(rx, rx+rw):
            if seed_mask[y, x] != 0:
                coordinates.append((y, x))

    # find the most distant x and y coordinates
    max_y = sorted(coordinates, reverse=True, key=lambda c: c[0])[0][0]
    min_y = sorted(coordinates, reverse=False, key=lambda c: c[0])[0][0]
    length_y = max_y-min_y

    max_x = sorted(coordinates, reverse=True, key=lambda c: c[1])[1][1]
    min_x = sorted(coordinates, reverse=False, key=lambda c: c[1])[1][1]
    length_x = max_x - min_x

    new_width = length_x * 6
    new_height = length_y * 6
    if new_width < 800:
        new_width = 800

    if new_height < 800:
        new_height = 800

    # shift the region at the centre
    start_x = int(new_width/2-length_x / 2)
    start_y = int(new_height/2-length_y / 2)

    delta_x = min_x - start_x
    delta_y = min_y - start_y

    converter = lambda c: ((c[0] - delta_y) % new_height, (c[1] - delta_x) % new_width)

    smaller_mask = np.zeros((new_height, new_width), np.uint8)
    for c in coordinates:
        smaller_mask.itemset(converter(c), 255)

    return (smaller_mask, converter)

def region_grow(image, seed_mask, max_MA):
    """Grows each region on the seed mask. Growing is finished when
    the minor axis of an ellipse around the region hits the limit.

    The algorithm works as follows:

    1) For each seed:
       i)   Initialize a minimum priority queue with the negatives of the seed pixels' 
            intensities
       ii)  Repeat until the priority queue is empty:
            1) Pop a point with the highest priority
            2) Find its neighbours outside the grown region
            3) For each neighbour:
               i)   Draw the neighbour on the seed mask
               ii)  Fit an ellipse to the seed and compute its minor axis
               iii) If the length of the minor axis is smaller than the limit
                    calculate the priority for the point: -0.5 * intensity + 0.5 * minor 
                    axis and add the point to the priority queue
               iv)  If the minor axis length violates the constraint remove the point from 
                    the seed mask.

    Parameters
    ----------
    image: numpy array
        Source image
    seed_mask: numpy array
        Seed regions. The algorithm is applied for each region.
    max_MA: int
        The maximum allowed minor axis of an ellipse fit to the growing region

    Returns
    -------
    list
        List of the grown regions' coordinates
    """

    start_time = timeit.default_timer()

    seeds = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds = seeds[0] if imutils.is_cv2() else seeds[1]

    grown_regions = []
    print("seed amount: %d " % len(seeds))

    #seeds = seeds[:20]
    for seed_i, seed in enumerate(seeds):
        grow_start_time = timeit.default_timer()
        # construct a list of coordinates
        seed_region = set([])
        (rx, ry, rw, rh) = cv2.boundingRect(seed)
        for y in range(ry, ry+rh):
            for x in range(rx, rx+rw):
                if seed_mask[y, x] != 0:
                    seed_region.add((y, x))

        # huge seed mask kills the performance. Draw the seed on a smaller mask.
        (smaller_mask, converter) = _resize_seed_mask(seed, seed_mask)

        height, width = image.shape[:2]

        # fill the heap with the intensities of the seed points
        neighbours = seed_region
        heap = []
        for n in neighbours:
            smaller_mask.itemset(converter(n), 255)
            intensity = image.item(n[0], n[1])
            heapq.heappush(heap, (-intensity, n))

        grown_pixels = set([])
        discarded = 0

        # do not add an already checked neighbour to the region
        checked = set([])
        while heap:
            priority, coordinates = heapq.heappop(heap)
            y = coordinates[0]
            x = coordinates[1]
            checked.add((y, x))

            connected_outside = [c for c in _get_connected_neighbours(y, x, width, height) if c not in seed_region
                                 and c not in grown_pixels and c not in checked]
            if not connected_outside:
                continue

            # check which of the outside pixels don't violate the minor axis limit and add
            # them to the priority queue for further processing
            for co in connected_outside:
                intensity = image.item(co[0], co[1])
                smaller_mask.itemset(converter(co), 255)

                new_seed_angle, new_seed_MA, new_seed_ma = fit_ellipse(smaller_mask)

                if new_seed_MA > max_MA:
                    discarded += 1
                    smaller_mask.itemset(converter(n), 0)
                else:
                    heapq.heappush(heap, (0.5*new_seed_MA + -0.5 * intensity, co))
                    grown_pixels.add(co)

            elapsed = timeit.default_timer() - grow_start_time
            if elapsed > 45:
                print("timeout occurred. Seed %d/%d" % (seed_i, len(seeds)))
                heap = []

        print("Seed size %d" % len(seed_region))
        print("New pixels %d" % len(grown_pixels))
        print("Discarded pixels %d" % discarded)

        grown_regions.append(list(grown_pixels) + list(seed_region))

    elapsed = timeit.default_timer() - start_time
    print("elapsed : %f (seeds %d)" %(elapsed, len(seeds)))
    return grown_regions


def open_by_reconstruction(image, iterations=1, ksize=3):
    """Morphological operation which removes small islands from the image
    but won't change the shape of those which remain after erosion.

    Dilated objects are prevented extending beyond the original shape
    by masking with the original image.

    Parameters
    ----------
    image: numpy array
        source image
    iterations: int
        number of times erosion is applied.
    ksize: int
        the size of structuring element used for erosion

    Returns
    -------
    numpy array
        opened image
    """

    eroded = cv2.erode(image, np.ones((ksize, ksize), np.uint8), iterations=iterations)

    this_iteration = eroded
    last_iteration = eroded
    while True:
        this_iteration = cv2.dilate(last_iteration, np.ones((ksize, ksize), np.uint8), iterations=1)
        this_iteration = this_iteration & image
        if np.array_equal(last_iteration, this_iteration):
            break
        last_iteration = this_iteration.copy()

    return this_iteration

def pca(image):
    """Performs principal component analysis on the image and reconstructs
    the image from the third principal component only. It produces an image on
    which the intensities of the weed pixels are emphasized.

    Parameters
    ----------
    image: numpy array
        original RGB image

    Returns
    -------
    numpy array
        an image reconstructed from its third principal component

    """

    height, width = image.shape[:2]

    # there are 3 channels in the rgb color space
    dimensions = 3

    mean, evecs = cv2.PCACompute(image.reshape((-1, dimensions)), mean=None)
    transformed = image.reshape((-1, dimensions)) * np.matrix([evecs[2]]).T

    transformed.shape = (height, width)
    transformed = -0.01 * transformed

    tr_u8 = (transformed.clip(0, 1) * 255).astype(np.uint8)

    return tr_u8

def fit_ellipse(region_mask):
    """Fits an ellipse to the region and returns its angle, minor axis and
    major axis. If the mask contains multiple seeds the largest is chosen.

    Parameters
    ----------
    region_mask: numpy mask
        a mask containing the seed region(s)

    Returns
    -------
    (float, float, float)
        angle, minor axis, major axis
    """

    cnts = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = cnts[0]

    # if there are multiple regions pick the largest
    # this situation might occur if growing process starts from seed points
    # apart each other
    if len(cnts) > 1:
        area = 0
        for cc in cnts:
            M = cv2.moments(cc)
            areac = int(M['m00'])
            if areac > area:
                c = cc
                area = areac

    (xx, yy), (MA, ma), angle = cv2.fitEllipse(c)
    return (angle, MA, ma)

def filter_regions(mask, ma_limit, ecc_limit, max_area=None, min_area=None):
    """Filters out the regions violating the given constraints.

    Parameters
    ----------
    mask: numpy array
        a mask containing regions
    ma_limit: float/int
        maximum allowed minor axis of an ellipse around a region
    ecc_limit: float/int
        maximum allowed eccentricity of an ellipse around a region
    area_limit: float/int
        maximum allowed area of a region

    Returns
    -------
    (int, numpy array)
        Amount of seeds on the new image and the new image
    """

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    filtered_regions = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
    i = 0
    for c in cnts:

        M = cv2.moments(c)
        #if not M:
        #    cv2.drawContours(filtered_regions,[c],-1, (255,255,255),cv2.FILLED)
        #    continue

        area = int(M['m00'])

        if max_area and area > max_area:
            continue
        if min_area and area < min_area:
            continue

        ecc1 = (M['nu20'] + M['nu02']) + sqrt((M['nu20'] - M['nu02'])**2 + 4*(M['nu11'])**2)
        ecc2 = 4 * (M['nu20'] * M['nu02'] - M['nu11']**2)

        # TODO why not ecc2? only one point?
        if not ecc2:
            continue

        (xx, yy), (MA, ma), angle = cv2.fitEllipse(c)

        ecc = ecc1/ecc2

        if ecc_limit:
            if ecc < ecc_limit:
                continue

        if ma_limit:
            if ma < ma_limit:
                continue

        i += 1
        cv2.drawContours(filtered_regions, [c], -1, (255, 255, 255), cv2.FILLED)
        #cv2.drawContours(filtered_regions, [c], -1, (255,255,255), 3)

    return (i, filtered_regions)

def draw_regions(roi_mask, image, draw_statistics, draw_rectangle, color=(0, 0, 255)):
    """Draws the contours of the rois on the original image.

    Parameters
    ----------
    roi_mask: numpy array
        a mask containing the rois
    image: numpy array
        the original image where the rois are marked
    draw_statistics: boolean
        if true ecc, area, minor axis length and major axis length are written on the image
    draw_rectangle: boolean
        if true rectangle is drawn instead of contours
    color: tuple, optional
        rgb color used to draw contours on the image. Default is red.
    """

    cnts = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        M = cv2.moments(c)
        if not M:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x+w+20, y+h+20), (0, 0, 255), 8)
            continue


        area = int(M['m00'])
        ecc1 = (M['nu20'] + M['nu02']) + sqrt((M['nu20'] - M['nu02'])**2 + 4*(M['nu11'])**2)
        ecc2 = 4 * (M['nu20'] * M['nu02'] - M['nu11']**2)
        if not ecc2:
            print("NO ECC2")
            ecc2 = 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x+w+20, y+h+20), (0, 0, 255), 3)
            continue

        ecc = ecc1/ecc2

        cY = int(M["m01"] / M["m00"])
        cX = int(M["m10"] / M["m00"])
        (xx, yy), (MA, ma), angle = cv2.fitEllipse(c)

        if draw_rectangle:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x+w+20, y+h+20), color, 6)
        else:
            color = (0, 0, 255)
            print("drawing contours")
            cv2.drawContours(image, [c], -1, color, 2)

        if draw_statistics:
            color = (0, 0, 0)
            cv2.putText(image, "ecc: %f" % ecc, (cX+20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            cv2.putText(image, "area: %d" % area, (cX+20, cY+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            cv2.putText(image, "MA: %d" % MA, (cX+20, cY+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            cv2.putText(image, "ma: %d" % ma, (cX+20, cY+75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

def search(filenames, constraints):
    """Detects common wild oats from the images. If a roi is detected its contours
    are drawn and the new image is saved into TARGET_DIR.

    Parameters
    ----------
    filenames: list of str
        the names of the image files
    constraints: list of dicts
        selection criteria for seed regions, pixels in growing process and final ROIs

        dicts contain the following keys:
        - seed_lower: lower intensity limit for seed pixels
        - seed_upper: upper intensity limit for seed pixels
        - seed_area: maximum allowed area for a seed
        - seed_length: maximum allowed length for a seed
        - seed_ecc: maximum allowed eccentricity for a seed

        - local_MA: maximum allowed length for a minor axis of an ellipse around the
                    growing region

        - grown_length: minimum length of a grown region
        - grown_ecc: minimum ecc of a grown region
    """

    # DEBUG
    #constraints = [constraints[2]]
    print('filenames: %d ' % len(filenames))
    for j, filename in enumerate(filenames):
        source = cv2.imread(filename)

        tr_u8 = pca(source)
        cv2.imwrite(TARGET_DIR + '/'+ 'pca.tif', tr_u8)

        found = 0
        for i, constraint in enumerate(constraints):

            # FIND SEED POINTS

            # GLOBAL THRESHOLD
            lower = constraint['seed_lower']
            upper = constraint['seed_upper']
            seeds2 = cv2.inRange(tr_u8, lower, upper)

            cv2.imwrite(TARGET_DIR + '/'+str(i) +'_seeds.tif', seeds2)

            # REMOVE NOISE AND SMALL ISLANDS
            img_opened2 = open_by_reconstruction(seeds2, 3, 3)
            cv2.imwrite(TARGET_DIR + '/'+'opened_seeds.tif', seeds2)

            # FILTER OUT SEEDS WHICH DON'T RESEMBLE A PART OF COMMON WILD OAT'S CULM
            min_ecc = constraint['seed_ecc']
            min_length = constraint['seed_length']
            max_area = constraint['seed_area']
            min_area = constraint.get('seed_area_min', None)
            (amount, filtered_seeds) = filter_regions(img_opened2, min_length, min_ecc, max_area, min_area)
            if amount > 450:
                print("Too many seeds(%d). Skipping the constraint." % amount)
                continue

            # LOCAL REGION GROWING
            cv2.imwrite(TARGET_DIR + '/'+ str(i) +'_filtered_seeds.tif', filtered_seeds)
            grown_regions = []
            max_MA = constraint['local_MA']
            grown_regions = region_grow(tr_u8, filtered_seeds, max_MA)

            #grown_regions = []
            # DRAW THE GROWN REGIONS ON A NEW MASK
            grown_seeds_mask = np.zeros([source.shape[0], source.shape[1], 1], np.uint8)
            for region in grown_regions:
                for coordinate in region:
                    grown_seeds_mask.itemset((coordinate[0], coordinate[1], 0), 255)

            # FILTER OUT THE REGIONS NOT SATISFYING THE SELECTION CRITERIA
            min_length = constraint['grown_length']
            min_ecc = constraint['grown_ecc']
            (amount, filtered_grown_regions) = filter_regions(grown_seeds_mask, min_length, min_ecc, None)
            found += amount

            # DRAW THE FINAL ROIS ON THE ORIGINAL IMAGE
            draw_regions(filtered_grown_regions, source, False, False)

        print('CONSTRAINTS CHECKED. AMOUNT OF FOUND ROIS %d ' % found)
        print('Image %d/%d processed' % (j+1, len(filenames)))

        if found:
            basename = filename.split('/')[len(filename.split('/'))-1].split('.')[0]
            filename = "%s/%s_rois.tif" % (TARGET_DIR, basename)
            print("image %s written" % filename)
            cv2.imwrite(filename, source)

    cv2.destroyAllWindows()


def main():
    """Initializes detection processing."""

    filenames = ["/home/makinen/utu/gradu/exported_oikea/%s" % f for f in os.listdir('/home/makinen/utu/gradu/exported_oikea') if f.endswith('tif')]

    filenames = ["/home/makinen/utu/gradu/exported_oikea/HI2A2575.tif"]
    def split(a, n):
        """Splits the given list into n parts"""
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    procs = 5   # Number of processes to create
    filename_chunks = list(split(filenames, procs))

    # Create a list of jobs and then iterate through
    # the number of processes appending each process to
    # the job list
    jobs = []

    constraints = []

    # SAMPLE 1 / HI2A2607.tif
    c = {}
    # constraints for selecting the seed areas
    c['seed_lower'] = 50
    c['seed_upper'] = 90
    c['seed_area'] = 2000
    c['seed_length'] = 60
    c['seed_ecc'] = 24

    # the maximum allower minor axis of a grown region
    c['local_MA'] = 50

    # selection criteria for final ROIs
    c['grown_length'] = 215
    c['grown_ecc'] = 13
    constraints.append(c)

    ##
    ### SAMPLE 2 / HI2A2606.tif
    c2 = {}
    # constraints for selecting the seed areas
    c2['seed_lower'] = 75
    c2['seed_upper'] = 120
    c2['seed_area_min'] = 300
    c2['seed_area'] = 1400
    c2['seed_length'] = 30
    c2['seed_ecc'] = 5

    # the maximum allower minor axis of a grown region
    c2['local_MA'] = 60

    # selection criteria for final ROIs
    c2['grown_length'] = 220
    c2['grown_ecc'] = 5
    constraints.append(c2)

    ##
    ### SAMPLE 3 / HI2A2633.tif
    c3 = {}
    c3['seed_lower'] = 60
    c3['seed_upper'] = 80
    c3['seed_area'] = 5000
    c3['seed_length'] = 62
    c3['seed_ecc'] = 15

    # the maximum allower minor axis of a grown region
    c3['local_MA'] = 75

    # selection criteria for final ROIs
    c3['grown_length'] = 400
    c3['grown_ecc'] = 10
    constraints.append(c3)

    ##
    #### SAMPLE 4 (1/2) / HI2A2632.tif
    c5 = {}
    c5['seed_lower'] = 90
    c5['seed_upper'] = 120
    c5['seed_area'] = 650
    c5['seed_length'] = 35
    c5['seed_ecc'] = 4

    # the maximum allower minor axis of a grown region
    c5['local_MA'] = 75

    # selection criteria for final ROIs
    c5['grown_length'] = 275
    c5['grown_ecc'] = 10
    constraints.append(c5)

    ##
    #### SAMPLE 4 (2/2) / HI2A2632.tif
    c6 = {}
    c6['seed_lower'] = 50
    c6['seed_upper'] = 65
    c6['seed_area'] = 2000
    c6['seed_length'] = 120
    c6['seed_ecc'] = 5

    # the maximum allower minor axis of a grown region
    c6['local_MA'] = 75

    # selection criteria for final ROIs
    c6['grown_length'] = 200
    c6['grown_ecc'] = 15
    constraints.append(c6)

    for i in range(0, procs):
        process = multiprocessing.Process(target=search, args=(filename_chunks[i], constraints))
        jobs.append(process)

    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()

    print("Processing complete.")

if __name__ == "__main__":
    main()
