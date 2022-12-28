# !bin/usr/python
import time
import datetime
import os
import time
import shutil
import numpy as np
from datetime import timezone, timedelta, datetime
import multiprocessing
from solarmeter import digitalize
from multiprocessing import Process

from time import sleep
from picamera2 import Picamera2
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

bucket = "AllSkyTUDD"
token = "MphEEMoaU_lXm_W3tGeSZ1knfZKMZKi_qJuh-w70G9KeBAJIpGjS2rK10TJjmrMU_aRN1jaSrxEWo4bYVOjhKQ=="
org = "paul.matteschk@ea-energie.de"
url = "https://us-central1-1.gcp.cloud2.influxdata.com"

client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

# from preproc import preproc

# https://stackoverflow.com/questions/41912594/how-to-speed-up-numpy-sum-and-python-for-loop
# operating directory for image saving


os.chdir("/home/pi/testenv/monster")

expos_ = []
images_ = []
Lux_ = []
Tcam_ = []


def capture_multiple_exposures(picam2, exp_list, callback):
    def match_exp(metadata, indexed_list):
        err_factor = 0.00001
        err_exp_offset = 33
        exp = metadata["ExposureTime"]
        gain = metadata["AnalogueGain"]
        for want in indexed_list:
            want_exp, _ = want
            if (
                    abs(gain - 1.0) < err_factor
                    and abs(exp - want_exp) < want_exp * err_factor + err_exp_offset
            ):
                return want
        return None

    indexed_list = [(exp, i) for i, exp in enumerate(exp_list)]
    while indexed_list:
        request = picam2.capture_request()
        match = match_exp(request.get_metadata(), indexed_list)
        if match is not None:
            indexed_list.remove(match)
            exp, i = match
            callback(i, exp, request)
        if indexed_list:
            exp, _ = indexed_list[0]
            picam2.set_controls(
                {
                    "ExposureTime": exp,
                    "AnalogueGain": 1.0,
                    "ColourGains": (1.0, 1.0),
                    "FrameDurationLimits": (25000, 50000),
                }
            )
            indexed_list.append(indexed_list.pop(0))
        request.release()


def callback_func(i, wanted_exp, request):
    print(i, "wanted", wanted_exp, "got", request.get_metadata()["ExposureTime"])
    expos_.append((i, request.get_metadata()["ExposureTime"]))
    images_.append((i, request.make_array("raw")))
    try:
        Lux_.append((i, request.get_metadata()["Lux"]))
        Tcam_.append((i, request.get_metadata()["SensorTemperature"]))
    except:
        pass


def metadata(ghi, refY):
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        lux = (
            influxdb_client.Point("CamMetadata")
                .field("Lux_0", Lux_[0][1])
                .field("Lux_1", Lux_[1][1])
                .field("Lux_2", Lux_[2][1])
                .field("Lux_3", Lux_[3][1])
                .field("Lux_4", Lux_[4][1])
                .field("Lux_5", Lux_[5][1])
                .field("Lux_6", Lux_[6][1])
                .field("Lux_7", Lux_[7][1])
                .field("Lux_8", Lux_[8][1])
                .field("Tcam_0", Tcam_[0][1])
                .field("Tcam_1", Tcam_[1][1])
                .field("Tcam_2", Tcam_[2][1])
                .field("Tcam_3", Tcam_[3][1])
                .field("Tcam_4", Tcam_[4][1])
                .field("Tcam_5", Tcam_[5][1])
                .field("Tcam_6", Tcam_[6][1])
                .field("Tcam_7", Tcam_[7][1])
                .field("Tcam_8", Tcam_[8][1])
        )
        solar = influxdb_client.Point("Globalstrahlung").field("GHI", ghi)
        Y = (
            influxdb_client.Point("refY")
                .field("refY0", refY[0])
                .field("refY1", refY[1])
                .field("refY2", refY[2])
                .field("refY3", refY[3])
                .field("refY4", refY[4])
                .field("refY5", refY[5])
                .field("refY6", refY[6])
                .field("refY7", refY[7])
                .field("refY8", refY[8])
        )
        write_api.write(bucket=bucket, org=org, record=[lux, solar, Y])
    except:
        pass


expos_r = [60, 106, 166, 273, 470, 773, 1258, 2032, 3291]
onefileo = np.array([], dtype="uint16")


def store_original(timestamp, images):
    images_r = [
        images[0][1],
        images[1][1],
        images[2][1],
        images[3][1],
        images[4][1],
        images[5][1],
        images[6][1],
        images[7][1],
        images[8][1],
    ]
    combined = np.asarray(images_r).view(np.uint16)
    print(combined.shape)
    np.save(f"{timestamp}_onefile.npy", combined)
    np.savez(
        f"{timestamp}.npz",
        images[0][1],
        images[1][1],
        images[2][1],
        images[3][1],
        images[4][1],
        images[5][1],
        images[6][1],
        images[7][1],
        images[8][1],
    )


def referenceY(timestamp, images):#[1::2, 0::2] rggb unten y nach rechts x # catch green
    try:
        referenceY0 = (np.average((images[0][1])[1::2, 0::2]) + np.average((images[0][1])[0::2, 1::2])) / 2 - 256
        referenceY1 = (np.average((images[1][1])[1::2, 0::2]) + np.average((images[1][1])[0::2, 1::2])) / 2 - 256
        referenceY2 = (np.average((images[2][1])[1::2, 0::2]) + np.average((images[2][1])[0::2, 1::2])) / 2 - 256
        referenceY3 = (np.average((images[3][1])[1::2, 0::2]) + np.average((images[3][1])[0::2, 1::2])) / 2 - 256
        referenceY4 = (np.average((images[4][1])[1::2, 0::2]) + np.average((images[4][1])[0::2, 1::2])) / 2 - 256
        referenceY5 = (np.average((images[5][1])[1::2, 0::2]) + np.average((images[5][1])[0::2, 1::2])) / 2 - 256
        referenceY6 = (np.average((images[6][1])[1::2, 0::2]) + np.average((images[6][1])[0::2, 1::2])) / 2 - 256
        referenceY7 = (np.average((images[7][1])[1::2, 0::2]) + np.average((images[7][1])[0::2, 1::2])) / 2 - 256
        referenceY8 = (np.average((images[8][1])[1::2, 0::2]) + np.average((images[8][1])[0::2, 1::2])) / 2 - 256


        refY = [
            referenceY0,
            referenceY1,
            referenceY2,
            referenceY3,
            referenceY4,
            referenceY5,
            referenceY6,
            referenceY7,
            referenceY8,
        ]
    except:
        pass

    return refY


def upload_original(timestamp):
    date = datetime.fromtimestamp(timestamp)
    YYYY = date.strftime("%Y")
    MM = date.strftime("%m")
    DD = date.strftime("%d")
    HH = date.strftime("%H")
    M = date.strftime("%M")
    S = date.strftime("%S")

    YYYYs = str(YYYY)
    MMs = str(MM)
    DDs = str(int(DD))

    shutil.move(
        f"/home/pi/testenv/monster/{timestamp}.npz",
        f"/home/pi/mnt/skydrive/DATASETS/TUDD/original/npz/{YYYYs}/{MMs}/{DDs}/{timestamp}.npz",
    )
    shutil.move(
        f"/home/pi/testenv/monster/{timestamp}_onefile.npy",
        f"/home/pi/mnt/skydrive/DATASETS/TUDD/original/onesky/{YYYYs}/{MMs}/{DDs}/{timestamp}_onefile.npy",
    )


def convert_rggb_to_rgb(rggb):  # RGGB (x,y) --> RGB = (x/2,y/2,3)

    ret_arr = np.zeros((3, rggb.shape[0] // 2, rggb.shape[1] // 2)).astype(np.uint16)
    blue = rggb[1::2, 1::2]  # blue
    green1 = rggb[0::2, 1::2]  # green
    green2 = rggb[1::2, 0::2]  # green
    red = rggb[0::2, 0::2]  # red

    ret_arr[0] = ret_arr[0] + red
    ret_arr[1] = ret_arr[1] + (green1 + green2) / 2
    ret_arr[2] = ret_arr[2] + blue

    return ret_arr  # RGB = (x/2,y/2,3)


def get_sky_area(radius, centre, image_name):
    data = np.load(image_name)
    resize_ration =
    centre = centre * resize_ratio
    radius = radius * resize_ratio
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
    avgR = np.mean(masked[:, :, 2])
    # avgG = np.mean(masked[:,:,1])
    # avgB = np.mean(masked[:,:,0])
    # print("Mean of channel R: ", avgR)
    # print("Mean of channel G: ", avgG)
    # print("MEan of channel B: ", avgB)

    # cv2.imshow(npy_9_imgs[0])
    # plt.imshow(sky)
    # plt.show()
    # cv2.imshow("mask", mask)
    # cv2.imwrite('sky_image.png', masked[:,:,2])
    # print("Saved Sky image as sky_image")
    # cv2.imshow("Mask applied to image", masked)
    # cv2.waitKey()
    return avgR

def resize_and_mask(img, width, height):
  w, h, _ = img.shape
  dims = (width, height)
  resized = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)
  return resized

#TESTSERIES FOR DIFFRENT RESOLUTIONS


def cap():
    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            raw={"format": "SRGGB12", "size": (2032, 1520)}, buffer_count=3
        )
        picam2.configure(config)
        picam2.start()

        exposure_list = [60, 120, 180, 300, 480, 780, 1260, 2040, 3300]
        # count = 1
        if True:
            # count = count + 1
            start = time.monotonic()
            images = images_
            expos = expos_  # real exposure times
            timestamp = int(datetime.now(timezone.utc).timestamp())
            
            ##################################################################
            
            capture_multiple_exposures(picam2, exposure_list, callback_func)                #CAPTURE 9 FRAMES RGGB12 (9,1520,2032)
            
            #################################################################

            metadata(digitalize(), referenceY(timestamp, images))                           #UPLOAD METADATA
            
            #################################################################
            
            images.sort(key=lambda tup: tup[0])
            expos.sort(key=lambda tup: tup[0])
            Lux_.sort(key=lambda tup: tup[0])
            Tcam_.sort(key=lambda tup: tup[0])

            store_original(timestamp, images)
            upload_original(timestamp)

            del expos_[:]
            del images_[:]
            del Lux_[:]
            del Tcam_[:]

            end = time.monotonic()
            print("Time:", end - start)

        picam2.stop()


t1 = time.perf_counter()
cap()

t2 = time.perf_counter()
print(f"Finished in {t2 - t1} seconds")
