import cv2
import numpy as np

class ViewPort():
    def __init__(self, max_width_, max_height_):
        self.max_width = max_width_
        self.max_height = max_height_
        #self.width = width_
        #self.height = height_
        #self.canvas = np.zeros((self.height, self.width, 3), np.uint8)    
        #self.max_zoom_factor = 1.0
        #self.min_zoom_factor = 0
        self.z_factor = 0

    def load(self, img):
        """
        self.z_factor = self.width/img.shape[1]
        z_height = img.shape[0] * self.z_factor
        if z_height > self.height:
            self.z_factor = self.height/img.shape[0]
        self.min_zoom_factor = self.z_factor
        """
        self.z_factor = self.max_width/img.shape[1]
        z_height = img.shape[0] * self.z_factor
        if z_height > self.max_height:
            self.z_factor = self.max_height/img.shape[0]
        self.min_zoom_factor = self.z_factor
        
        resized_img = cv2.resize(img, None, fx=self.z_factor, fy=self.z_factor)
        return resized_img

    """
    def zoom(self, img):
        #global resized_img, x_pan_max, y_pan_max
        resized_img = cv2.resize(img, None, fx=self.z_factor, fy=self.z_factor)
        #x_pan_max = resized_img.shape[1] - window_width
        #y_pan_max = resized_img.shape[0] - window_height
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (i < resized_img.shape[0] and j < resized_img.shape[1]):
                    self.canvas[i,j] = resized_img[i, j]
                else:
                    self.canvas[i, j] = (0,0,0)
        return self.canvas
    """

"""
x_pan_max = 0.0
y_pan_max = 0.0
x_offset = 0
y_offset = 0



def pan():
    
    for i in range(0, window_height):
        for j in range(0, window_width):
            if (i < resized_img.shape[1] and j < resized_img.shape[0]):
                if (i+y_offset < resized_img.shape[1], j+x_offset < resized_img.shape[0]):
                    canvas[i,j] = resized_img[i + int(y_offset * y_pan_max) , j + int(x_offset * x_pan_max)]
            else:
                canvas[i, j] = (0,0,0)
    cv2.imshow("img", canvas)



    while(1):
        key =cv2.waitKey(0)
        if key & 0xFF == ord("z"):
            z_factor += 0.1
            if z_factor <= max_zoom_factor:
                zoom(z_factor)
            else:
                z_factor -= 0.1
        if key & 0xFF == ord("x"):
            z_factor -= 0.01
            if z_factor >= min_zoom_factor:
                zoom(z_factor)
            else:
                z_factor += 0.1
        
        if key & 0xFF == ord("d"):
            x_offset += 0.1
            if x_offset > 1:
                x_offset = 1
            pan()
        if key & 0xFF == ord("a"):
            x_offset -= 0.1
            if x_offset < 0:
                x_offset = 0
            pan()
        if key & 0xFF == ord("w"):
            y_offset += 0.1
            if y_offset > 1:
                y_offset = 1
            pan()
        if key & 0xFF == ord("s"):
            y_offset -= 0.1
            if y_offset < 0:
                y_offset = 0
            pan()

        if key & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
"""