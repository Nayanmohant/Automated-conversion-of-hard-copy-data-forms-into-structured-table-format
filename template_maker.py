import cv2
import numpy as np

class PointsHolder:
    
    def __init__(self):
        self.points = []

    def insert(self, point):
        self.points.append(point)

    def select(self, index):
        return self.points[index]
    

class Template:
    def __init__(self, img_, thickness_, color_):
        self.img = img_
        self.width = img_.shape[1]
        self.height =img_.shape[0]
        self.thickness = thickness_
        self.color = color_
        self.points_holder = PointsHolder()
        self.pt1 = None
        self.pt2  = None
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.window_name = None
        self.size = None

    def draw_rectangle(self, points):
        #cv2.rectangle(self.canvas, points[0], points[1], self.color, self.thickness)
        cv2.rectangle(self.img, points[0], points[1], self.color, self.thickness)
        cv2.imshow(self.window_name, self.img)
        #return self.img
        #return self.canvas
    
    def mark_fields(self, window_name_):
        #points = rectangle_points()
        self.window_name = window_name_
        cv2.setMouseCallback(self.window_name, get_selection, self)
    
    def scale_points(self, img, scale_factor_x, scale_factor_y):
        scaled_points = []
        
        for point in self.points_holder.points:
            pt1 = [0, 0]
            pt2 = [0, 0]
            pt1[0] = int(point[0][0] * scale_factor_y)
            pt1[1] = int(point[0][1] * scale_factor_x)
            pt2[0] = int(point[1][0] * scale_factor_y)
            pt2[1] = int(point[1][1] * scale_factor_x)
            scaled_points.append([pt1, pt2])
            #cv2.rectangle(img, pt1, pt2, self.color, self.thickness)
        print("scaled points")
        print(scaled_points)
        self.points_holder.points = scaled_points
        return img


def get_selection(event, x, y, flags, self):

    if event == cv2.EVENT_LBUTTONDOWN:
        self.pt1 = (x,y)
    if event == cv2.EVENT_LBUTTONUP:
        self.pt2 = (x,y)
        self.points_holder.insert([self.pt1, self.pt2])
        self.draw_rectangle(self.points_holder.points[-1])

"""
if __name__ == '__main__':

    pts = points_holder()

    pt = ((1,2),(4,5))

    pts.insert(pt)

    print( pts.select(0))
"""