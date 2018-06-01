A script which detects common wild oats from aerial oat field images.
The algorithm is based on region growing which adds to the
growing region its neighbour pixels if they satisfy the membership criterion.

In common wild oat detection membership criterion needs to minimize
the width of a region and maximize the intensity of pixels with certain weights
to make the suspected region to grow along the culm as far as possible if it is
a common wild oat. After growing, regions can be classified into different 
species according to their length.

Before applying the growing algorithm images are preprocessed with
principal component analysis which emphasizes green hue and
makes the color difference of weeds and background larger.
