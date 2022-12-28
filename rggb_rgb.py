#RESIZE AND MASK
import glob
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import numpy as np
from numpy import asarray, save
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt


def convert_rggb_to_rgb(rggb): #RGGB (x,y) --> RGB = (x/2,y/2,3)

 ret_arr = np.zeros((3,rggb.shape[0]//2,rggb.shape[1]//2)).astype(np.uint16)
 blue = rggb[1::2, 1::2]    # blue
 green1 = rggb[0::2, 1::2]  # green
 green2 = rggb[1::2, 0::2]  # green
 red = rggb[0::2, 0::2]     # red

 ret_arr[0] = ret_arr[0] + red
 ret_arr[1] = ret_arr[1] + (green1 + green2)/2
 ret_arr[2] = ret_arr[2] + blue

 return ret_arr #RGB = (x/2,y/2,3)


def get_sky_area(radius, centre, image_name):
    data = np.load(image_name)

    # create Pillow image
    image2 = Image.fromarray(np.asarray(data[0]).astype(np.uint16))
    print(type(image2))
    cv2_img = cv2.cvtColor(np.array(image2), cv2.COLOR_GRAY2BGR)
    
    sky = cv2_img.copy()
    # create a black image with the same size as our 
    # image that contains the moon, we then create
    # a white circle on the black image
    mask = np.zeros(sky.shape[:2], dtype="uint16")
    cv2.circle(mask, centre, radius, 255, -1)
    # apply the mask to our image
    masked = cv2.bitwise_and(sky, sky, mask=mask)
    avgR = np.mean(masked[:,:,2])
    #avgG = np.mean(masked[:,:,1])
    #avgB = np.mean(masked[:,:,0])
    #print("Mean of channel R: ", avgR)
    #print("Mean of channel G: ", avgG)
    #print("MEan of channel B: ", avgB)
    
    #cv2.imshow(npy_9_imgs[0])
    #plt.imshow(sky)
    #plt.show()
    #cv2.imshow("mask", mask)
    #cv2.imwrite('sky_image.png', masked[:,:,2])
    #print("Saved Sky image as sky_image")
    #cv2.imshow("Mask applied to image", masked)
    #cv2.waitKey()
    return avgR
    
CENTRE= (1024, 804)
RADIUS = 1070
IMAGE_NAME = "1671709803_onefile.npy"
avg0 = get_sky_area(radius = RADIUS, centre = CENTRE, image_name=IMAGE_NAME)
