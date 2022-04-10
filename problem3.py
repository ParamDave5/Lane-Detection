import numpy as np
import matplotlib.pyplot as plt
import cv2

def leastsquare(points):
    x = []
    y = []
    #y = ax^2 + bx + c
    for point in points:
        x.append([point[1]**2 , point[1],1])
        y.append(point[0])
    
    x = np.array(x)
    y = np.array(y)

    xtx = np.dot(x.T , x)
    try:
        xt_x_inv = np.linalg.pinv(np.dot(x.T,x))
        xty = np.dot(x.T,x)
        answer = np.dot(xt_x_inv , xty)
        return answer
    except:
        return []

def yellow_mask(image):
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    lower = np.array([22,100,100])
    upper = np.array([45 ,255,255])
    yellow_mask = cv2.inRange(hsv , lower , upper)
    return yellow_mask

def white_mask(image):
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    lower = np.array([0,0,230])
    upper = np.array([255 ,255,255])
    white = cv2.inRange(hsv,lower , upper)
    return white
#roi for yello and white mask only
def roi(img , vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask , vertices , match_mask_color)
    masked_image = cv2.bitwise_and(img , mask)
    return masked_image

def warp(img , src , dst):
    height , width = img.shape[:2]
    forward_warp = cv2.getPerspectiveTransform(src,dst)
    inverse_warp = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(img , forward_warp , (width,height) , flags = cv2.INTER_LINEAR)
    return warped , forward_warp , inverse_warp

def unwarp(img):
    
    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)
    
    return unwarped

def lane(frame ,left , right ):
    plt.imshow(out_img)
    plt.plot(left, ploty, color='yellow')
    plt.plot(right, ploty, color='yellow')

def sliding_window(img):
    ym_per_pix = 3.048/100 
    xm_per_pix = 3.7/378 

    nwindows = 9
    margin = 100
    minpix = 50
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
 
    out_img = np.dstack((img, img, img))*255
 
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int32(img.shape[0]/nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img
   

def get_center_dist(leftLine, rightLine):
    ym_per_pix = 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
    xm_per_pix = 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
    
    y = 700.
    image_center = 640. * xm_per_pix
    
    leftPos = leftLine[0]*(y**2) + leftLine[1]*y + leftLine[2]
    rightPos = rightLine[0]*(y**2) + rightLine[1]*y + rightLine[2]
    lane_middle = int((rightPos - leftPos)/2.)+leftPos
    lane_middle = lane_middle * xm_per_pix
    
    mag = lane_middle - image_center
    if (mag > 0):
        head = "Right"
    else:
        head = "Left"
            
    return head, mag

def combine_radii(leftLine, rightLine):
    ym_per_pix = 3.048/100
    y_eval = 720. * ym_per_pix
    left = leftLine
    right = rightLine
    curve_rad_left = ((1 + (2*left[0]*y_eval + left[1])**2)**1.5) / np.absolute(2*left[0])
    curve_rad_right = ((1 + (2*right[0]*y_eval + right[1])**2)**1.5) / np.absolute(2*right[0])
    return curve_rad_left, curve_rad_right ,  np.average([curve_rad_left, curve_rad_right])
    
def create_final_image(img, binary_warped, leftLine, rightLine, show_images=False):
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=20)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=20)
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    if show_images:
        plt.figure(figsize=(9,9))
        plt.imshow(color_warp)
        
        plt.figure(figsize=(9,9))
        plt.imshow(result)
    
    return result

def add_image_text(img, radius1 , radius2,average, head, center):
    
    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_DUPLEX
    
    text = 'Radius of curvature for left lane: ' + '{:04.0f}'.format(radius1) + 'm'
    cv2.putText(img, text, (50,100), font, 1, (0,255, 0), 2, cv2.LINE_AA)

    text = 'Radius of curvature for right lane: ' + '{:04.0f}'.format(radius2) + 'm'
    cv2.putText(img, text, (50,150), font, 1, (0,255, 0), 2, cv2.LINE_AA)

    text = 'Average Radius of curvature: ' + '{:04.0f}'.format(int(average)) + 'm'
    cv2.putText(img, text, (50,200), font, 1, (0,255, 0), 2, cv2.LINE_AA)

    text = '{:03.2f}'.format(abs(center)) + 'm '+ head + ' of center'
    cv2.putText(img, text, (50,255), font, 1, (0,255, 0), 2, cv2.LINE_AA)
    
    if (radius1 - radius2) > 300:
        text = 'Turn Right ' 
        cv2.putText(img, text, (50,300), font, 1, (0,255, 0), 2, cv2.LINE_AA)
    if (radius2 - radius2) > 300:
        text = 'Turn Left ' 
        cv2.putText(img, text, (50,300), font, 1, (0,255, 0), 2, cv2.LINE_AA)
    elif (abs(radius1 - radius2)) < 300:
        text = 'Go Straight ' 
        cv2.putText(img, text, (50,300), font, 1, (0,255, 0), 2, cv2.LINE_AA)
    return img

def final_pipeline(img , warped):
    image_og = img.copy()
    left_fit, right_fit, left_fit_m, right_fit_m, out_img = sliding_window(warped)
    left_rad , right_rad , curve_rad = combine_radii(left_fit, right_fit)
    head, center = get_center_dist(left_fit, right_fit)
    result = create_final_image(img, warped, left_fit, right_fit)   
    result = add_image_text(result, left_rad , right_rad ,curve_rad , head, center)
    return result

def drawcircle(frame , x , y ):
    for i in range(len(x)):
        frame = cv2.circle(frame, (int(x[i]),int(y[i])), radius=0, color=(0, 255, 255), thickness=-1)
    return frame

def concatenated(img1, img2, img3, img4 ,img5):
    
    img1 = cv2.resize(img1, (950,550), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (310,180), interpolation = cv2.INTER_AREA)
    img3 = cv2.resize(img3, (310,180), interpolation = cv2.INTER_AREA)
    img4 = cv2.resize(img4, (310,370), interpolation = cv2.INTER_AREA)
    img5 = cv2.resize(img5, (310,370), interpolation = cv2.INTER_AREA)
    result1 = np.concatenate((img2, img3), axis = 1)
    result2 = np.concatenate((img4, img5), axis = 1)
    result3 = np.concatenate(( result1,result2))
    result4 = np.concatenate((img1, result3), axis = 1)
 
    return result1, result2, result3, result4 


cap = cv2.VideoCapture('challenge.mp4')

# frame_height = int(550)
# frame_width = int(1570)

# size = (frame_width, frame_height)

# results = cv2.VideoWriter('problem3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while(True):
    ret , frame = cap.read()
    
    if ret == True:
        og = frame.copy()
        
        height = int(frame.shape[0])
        width = int(frame.shape[1])
        roi_vertices = [(0,height) , (int(width/2)  , int(height/2)) , (width,height)]
        frame = roi(frame , np.array([roi_vertices] ,np.int32))
        yellow_m = yellow_mask(frame)

        white_m  = white_mask(frame)
        #frame to combine
        combine = cv2.bitwise_or(yellow_m,white_m)
        
        #warping
        src = np.float32([(575,464) ,(707,464) , (258,682) , (1049,682) ])
        dst = np.float32([ (400,0) ,(width-400 ,0) ,(400,height) , (width-400,height)])
        #one frame to combine
        warp_ , M , M_inv = warp(combine , src , dst)

        hist = sliding_window(warp_)
        left_fit, right_fit, left_fit_m, right_fit_m, out_img = sliding_window(warp_)

        ploty = np.linspace(0 , warp_.shape[0]-1 , warp_.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0] *ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        result = final_pipeline(og , warp_)
        combine = np.stack((combine,)*3, axis=-1)
        warp_ = np.stack((warp_,)*3, axis=-1)
        #image to concatenate
        lines = drawcircle(out_img , left_fitx , ploty)
        lines = drawcircle(lines , right_fitx , ploty) 
        result1 , result2 , result3 ,  result4  = concatenated(result , og,combine ,warp_ , lines)
        cv2.imshow('frame' , result4)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break
    else:
        break

# results.release()

cap.release()
cv2.destroyAllWindows

