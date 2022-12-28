#capturing exposure timeseries


from picamera2 import Picamera2

def capture_multiple_exposures(picam2, exp_list, callback):

    def match_exp(metadata, indexed_list):
        err_factor = 0.05
        err_exp_offset = 50
        exp = metadata["ExposureTime"]
        gain = metadata["AnalogueGain"]
        for want in indexed_list:
            want_exp, _ = want
            if abs(gain - 1.0) < err_factor and abs(exp - want_exp) < want_exp * err_factor + err_exp_offset:
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
            picam2.set_controls({"ExposureTime": exp, "AnalogueGain": 1.0})
            indexed_list.append(indexed_list.pop(0))
        request.release()

def callback_func(i, wanted_exp, request):
    print(i, "wanted", wanted_exp, "got", request.get_metadata()["ExposureTime"])
    images.append((i, request.make_array("raw"))) #"main"


with Picamera2() as picam2:
    config = picam2.create_preview_configuration(raw={})
    picam2.configure(config)
    picam2.start()

    images = []
    exposure_list = [1000, 2000, 3000, 4000]
    capture_multiple_exposures(picam2, exposure_list, callback_func)

    picam2.stop()
