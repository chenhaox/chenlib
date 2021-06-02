import format
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from src.cv_orbbec import Orbbec_DpethCamera
from src.cv_opencv import RGB_Cam
def set_camera(img, camera:type):
    img.cam_pointer = camera
    return img

class Img(object):
    def __init__(self, cam_pointer):
        self.cam_pointer = cam_pointer
        if self.cam_pointer is None:
            raise ModuleNotFoundError

    def get_rgb_image(self):
        img = self.cam_pointer.read()[1]
        return img

    def get_depth_image(self):
        img = self.cam_pointer.read()[0]
        if np.max(img) > 255:   #generalized
            img = img/np.amax(img)*255
        img = img.astype(np.uint8)
        img = cv.applyColorMap(img, cv.COLORMAP_JET)
        return img

    def get_image(self, image_type):
        assert image_type in ["rgb","depth", "both"]
        if image_type == "rgb" : return self.get_rgb_image()
        if image_type == "depth" : return self.get_depth_image()
        if image_type == "both": return np.concatenate((self.get_rgb_image(),self.get_depth_image()))

    def show_video(self,process_callback = None, video_type = "rgb"):
        if process_callback is None:
            process_callback = lambda img: img
        print("Press \"q\" to quit showing video")
        print("Press \"k\" to save a frame")
        while (True):
            # Capture the video frame
            # by frame
            frame = process_callback(self.get_image(image_type= video_type))
            # Display the resulting frame
            cv.imshow('frame', frame)
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv.waitKey(20) & 0xFF == ord('q'):
                break
            if cv.waitKey(100) & 0xFF == ord('k'):
                format_name = format.time_format()
                self.save_rgb_img(f"/home/hlabbunri/Desktop/chenhao/训练数据/images/val/{format_name}.jpg")
                print(f"pic {format_name} created")
                continue
        # Destroy all the windows
        cv.destroyAllWindows()

    def save_rgb_img(self, path, img = None):
        if img is None:
            img = self.get_rgb_image()
        cv.imwrite(path,img)

    def save_depth_img(self, path, img = None):
        if img is None:
            img = self.get_depth_image()
        cv.imwrite(path,img)

class ImgProcessor(object):
    pass

if __name__ == "__main__":
    img_stream = Img(cam_pointer=Orbbec_DpethCamera)
    # img_stream2 = Img(cam_pointer=RGB_Cam)
    # for i in range(20):
    # image =  Img.get_rgb_image()
    img_depth = img_stream.show_video(video_type="rgb")
