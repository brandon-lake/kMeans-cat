""" Program which uses KMeans clustering to simplify the album art from one of my favourite albums into just a few colours, then apply this colour palatte to a photo I took of my cat

@author: Brandon Lake
"""
import numpy as np
from skimage import data
from skimage import io
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

## Read in image which will train the system
album_art = io.imread("culprate_others.jpg")
album_shape = album_art.shape
flat_album = album_art.reshape(album_shape[0] * album_shape[1], album_shape[2])
plt.figure()
plt.title("Cuplrate - Others (album)")
plt.imshow(album_art)
plt.axis('off')
plt.show()

inertias = []
x_axis = []
for x in range (4, 12):
    km = KMeans(n_clusters=x, n_init=5)
    km.fit(flat_album)
    inertias.append(km.inertia_)
    x_axis.append(x)

    # create new picture from x cluster points
    new_picture = np.round(km.cluster_centers_[km.predict(flat_album)]).astype(np.int)
    # reshape image back
    new_picture = new_picture.reshape(album_shape[0], album_shape[1], album_shape[2])

    # display image
    plt.figure()
    plt.title(f"Album - k = {x}")
    plt.imshow(new_picture)
    plt.axis('off')
    plt.show()

# display graph of k values vs inertia for 2 - 20
plt.figure()
plt.title("Inertia vs K-value")
plt.plot(x_axis, inertias, marker=".")
plt.show()


## Read in picture of my cat
cat = io.imread("Rosie.jpg")
cat_shape = cat.shape
flat_cat = cat.reshape(cat_shape[0] * cat_shape[1], cat_shape[2])
plt.figure()
plt.title("My Cat")
plt.imshow(cat)
plt.axis('off')
plt.show()

k_to_try = [6, 8, 12]

for x in range(len(k_to_try)):
    # predict with the k cluster system, with my chosen k value
    km = KMeans(n_clusters=k_to_try[x])
    km.fit(flat_album)
    new_cat = np.round(km.cluster_centers_[km.predict(flat_cat)]).astype(np.int)

    # reshape cat back
    new_cat = new_cat.reshape(cat_shape[0], cat_shape[1], cat_shape[2])

    # display new cat
    plt.figure()
    plt.title(f"My Cat - {k_to_try[x]} Clusters")
    plt.imshow(new_cat)
    plt.axis('off')
    plt.show()
