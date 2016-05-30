#handle = vot.VOT("rectangle")
import vot
import sys
import time
import cv2
import numpy
import collections
import flow
import simulator
import ORF
from ncc import NCCTracker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
handle = simulator.simulator("/home/boka/arp/david/")
selection = handle.region()

imagefile = handle.frame()
print("prvo")
if not imagefile:
    sys.exit(0)

#image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(imagefile, cv2.IMREAD_COLOR)
print(imagefile)
tracker = NCCTracker(image, selection)
tracker_flow = flow.flow(image, selection)
tracker_OT = ORF.flow(image, selection)
print("do tukej")
plt.ion()
plt.figure()
while True:
    imagefile = handle.frame()
    if not imagefile:
        break

    image = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    region = tracker.track(image)
    regionOrg = tracker_OT.track(image)
    plt.clf()
    a = plt.imshow(image)
    #tracker_flow.set_region(tracker.position)
    region_flow = tracker_flow.track(image)

    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((region.x, region.y), region.width, region.height, fill=None, alpha=1, color='yellow'))
    currentAxis.add_patch(
        Rectangle((regionOrg.x, regionOrg.y), regionOrg.width, regionOrg.height, fill=None, alpha=1, color='green'))
    currentAxis.add_patch(
        Rectangle((region_flow.x, region_flow.y), region_flow.width, region_flow.height, fill=None, alpha=1, color='red'))

    plt.draw()
    handle.report(region)
    time.sleep(0.1)
plt.show()