import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import os
ROOT = os.path.abspath(os.path.dirname(__file__))
# camera configuration
REDIST_PATH = f"{ROOT}/Redist_orbbec"
PIXEL_FORMAT = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM
# if use USB2.0, pixel=np.array([400, 640])
PIXEL_SIZE_X = 640
PIXEL_SIZE_Y = 480
FPS = 30

def StreamStart():
    openni2.initialize(REDIST_PATH)
    dev = openni2.Device.open_any()
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    color_stream = dev.create_color_stream()
    color_stream.start()
    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=PIXEL_FORMAT, resolutionX=PIXEL_SIZE_X,
                           resolutionY=PIXEL_SIZE_Y, fps=FPS))
    dev.set_depth_color_sync_enabled(True)
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    return depth_stream, color_stream

class Orbbec_DpethCamera(object):
    depth_stream, color_stream = StreamStart()

    @classmethod
    def read_depth(cls, pixel=np.array([480, 640])):
        frame = cls.depth_stream.read_frame()
        frame_dpt = cv2.flip(np.array(frame.get_buffer_as_uint16()).reshape(pixel), 1)
        return frame_dpt

    @classmethod
    def read_rgb(cls, pixel=np.array([480, 640, 3])):
        frame = cls.color_stream.read_frame()
        frame_rgb = cv2.flip(np.array(frame.get_buffer_as_uint8()).reshape(pixel), 1)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        return frame_rgb

    @classmethod
    def read(cls):
        return cls.read_depth(), cls.read_rgb()


