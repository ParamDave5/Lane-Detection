from turtle import position
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
# from skimage.transform import probabilistic_hough_line
# cap = cv2.VideoCapture('challenge.mp4')
cap = cv2.VideoCapture('whiteline.mp4')
previous = []
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
#returns m and c
def line(x1,y1,x2,y2):
    m = (y2 - y1)/(x2-x1)
    c = y2 - (m*x2)
    return m,c

def avg(line):
    m = 0
    c = 0
    for i in line:
        m += i[0]
        c += i[1]
    m = m/len(line)
    c = c/len(line)
    return m,c

def roi(img , vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask , vertices , match_mask_color)
    masked_image = cv2.bitwise_and(img , mask)
    return masked_image

def where(mask):
    where = 0
    half = int(len(mask)/2)
    left = mask[: , :half]
    right = mask[:,half:]
    left_white = np.where(left == 0)
    right_white = np.where(right == 0)
    left_len = left_white[0].shape
    right_len = right_white[0].shape
    if left_len > right_len:
        where = -1
    else:
        where = 1
    
    return where

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width , frame_height)

# result = cv2.VideoWriter('lane_detection_non.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while(cap.isOpened()):
    ret , frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        # frame = cv2.flip(frame,1)
        ret , thresh = cv2.threshold(gray , 180,255,cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh , (5,5) ,0 )
        height = frame.shape[0]
        width = frame.shape[1]

        roi_vertices = [(0,height) , (width/2  , height/2) , (width,height)]

        cropped_frame = roi(frame , np.array([roi_vertices] ,np.int32))
        gray = cv2.cvtColor(cropped_frame , cv2.COLOR_BGR2GRAY)
        ret , thresh = cv2.threshold(gray , 180,255,cv2.THRESH_BINARY)

        #detect yellow color
        hsv = cv2.cvtColor(cropped_frame , cv2.COLOR_BGR2HSV)
        lower = np.array([22,100,100])
        upper = np.array([45 ,255,255])
        yellow_mask = cv2.inRange(hsv , lower , upper)
        position = where(yellow_mask)

        kernel = np.ones((5,5) , np.uint8)
        dilation = cv2.dilate(yellow_mask , kernel ,iterations = 1 )
        
        wlower = np.array([0,0,230])
        wupper = np.array([255 ,255,255])
        white = cv2.inRange(hsv,wlower , wupper)
        
        bit = cv2.bitwise_or(white , yellow_mask)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(bit,kernel,iterations = 3)

        edges = cv2.Canny(bit , 280,360 , apertureSize= 3)
        lines = cv2.HoughLinesP(edges , 1 , np.pi/180 , threshold = 50 , minLineLength = 5 , maxLineGap = 100)
        lines_list = []
        mc = []
        for point in lines:
            x1 , y1 , x2 , y2 = point[0]
            m,c = line(x1,y1,x2,y2)
            mc.append([round(m,3),round(c,3)])
        a = []
        b = []
        final = []
    
        try:
            for i in mc:
                if i[0] < 0:
                    a.append(i)
                else:
                    b.append(i)
            m1,c1 = avg(a)
            m2,c2 = avg(b)
            final = [[m1,c1] , [m2,c2]]     
            previous = final
        except:
            final = previous   
       
        lines = []
        for points in final:
           
            if position == -1 :
                #green
                x11 = 0 
                y11 = 255
                z11 =  0
                #red
                x = 0
                y = 0
                z = 255
                y1 = width 
                x1 = int((y1 - points[1])/points[0]) 
                if points[1] < 150:
                    y2 = int(3*height/5)
                    x2 = int((y2 - points[1])/points[0])
                    
                else:
                    y2 = int(3*height/5)
                    x2 = int((y2 - points[1])/points[0])
                if points[1] >150:
                    cv2.line(frame , (x1,y1) , (x2,y2) , (x,y,z),3)
                else:
                    cv2.line(frame , (x1,y1) , (x2,y2) , (x11,y11,z11),3)
                lines.append([x1,y1])
                lines.append([x2,y2])
            if position == 1:
                x = 0 
                y = 255
                z =  0
                x11 = 0
                y11 = 0
                z11 = 255
                y1 = width 
                x1 = int((y1 - points[1])/points[0]) 
                if points[1] < 150:
                    y2 = int(3*height/5)
                    x2 = int((y2 - points[1])/points[0])
                else:
                    y2 = int(3*height/5)
                    x2 = int((y2 - points[1])/points[0])
                if points[1] >100:
                    cv2.line(frame , (x1,y1) , (x2,y2) , (x,y,z),3)
                else:
                    cv2.line(frame , (x1,y1) , (x2,y2) , (x11,y11,z11),3)
                lines.append([x1,y1])
                lines.append([x2,y2])
            lines_list.append([(x1,y1) , (x2,y2)])
        cv2.line(frame , lines[1] , lines[3] , (0,0,0),3)

        cv2.imshow('Frame',frame)
        # result.write(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# result.release()
cap.release()
cv2.destroyAllWindows




