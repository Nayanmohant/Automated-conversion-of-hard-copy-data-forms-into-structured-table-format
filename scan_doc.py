import cv2
import numpy as np


def order_points(pts):
    '''Rearrange coordinates to order:
       top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()


def scan(img):
    og_img = img.copy()

    # Resize image to workable size
    dim_limit = 720
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

    # Create a copy of resized original image for later use
    scaled_img_copy = img.copy()

    scale_f_x = og_img.shape[0]/img.shape[0]
    scale_f_y = og_img.shape[1]/img.shape[1]   

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Detecting Edges through Contour approximation
    if len(page) == 0:
        return scaled_img_copy
    # loop over the contours
    for c in page:
        # approximate the contour
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # if our approximated contour has four points
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    # Rearranging the order of the corner points.
    corners = order_points(corners)

    # Finding Destination Co-ordinates
    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    # Finding the maximum width.
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    # Finding the maximum height.
    h = max(int(h1), int(h2))

    # Final destination co-ordinates.
    destination_corners = order_points(np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]))

    #print("corners", corners)
    #print("destination corners", destination_corners)
    #for i in range (0,4):    
    #    cv2.circle(scaled_img_copy, corners[i], 2, (0,255,0), 2)
    #    cv2.circle(scaled_img_copy, destination_corners[i], 2, (0,0,255), 2)
    #cv2.imwrite('projects/project1/scanned/scaled_corners.jpg', scaled_img_copy)
    
    for i in range (0,4):
        corners[i][0] = int(corners[i][0] * scale_f_x)
        corners[i][1] = int(corners[i][1] * scale_f_y)
        destination_corners[i][0] = int(destination_corners[i][0] * scale_f_x)
        destination_corners[i][1] = int(destination_corners[i][1] * scale_f_y)
        #cv2.circle(og_img, corners[i], 5, (0,255,0), 5)
        #cv2.circle(og_img, destination_corners[i], 5, (0,0,255), 5)
    #cv2.imwrite('projects/project1/scanned/og_corners.jpg', og_img)
    
    h, w = og_img.shape[:2]
    # Getting the homography.
    homography, mask = cv2.findHomography(np.float32(corners), np.float32(destination_corners), method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
    
    # Perspective transform using homography.
    un_warped = cv2.warpPerspective(og_img, np.float32(homography), (w, h), flags=cv2.INTER_LINEAR)

    # Crop
    final = un_warped[:destination_corners[2][1], :destination_corners[2][0]]
    
    return final