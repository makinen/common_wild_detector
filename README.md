A script which detects common wild oats from aerial images of oat fields.
The algorithm is based on region growing which adds to the growing region its 
neighbour pixels if they satisfy a membership criterion.

The membership criterion needed to segment common wild oats minimizes the width of a
region and maximizes the intensity of pixels with certain weights. It guides the
algorithm to grow the region along the culm of a plant as far as possible and makes
it possible to classify the grown regions into different species according to their lenght.

Before growing the images are preprocessed with principal component analysis which 
emphasizes green hue and makes the color difference of weeds and background larger.
