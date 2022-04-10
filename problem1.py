import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import argparse
#--Method 1 for HE 
#--Method 2 for AHE
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Method', default= '1', help='Method to perform Histogram equalization')
args = parser.parse_args()
method = args.Method

#contains all the 25 images from the folder adaptive_hist_data
images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

#reads a single image
def read(path):
    img = cv2.imread(path)
    return img

#creates a final image using histgram equalization
def channel(b):
    b_ = b.copy
    hist , bins = np.histogram(b.flatten() , 256 , [0,256])
    cdf = hist.cumsum()
    cdf_norm = cdf*hist.max()/cdf.max()

    cdf_mask = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max()- cdf_mask.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[b]
        # plt.imshow(img2)
        # plt.show()
    return img2

#main histgram equalization function to to split the images , perform the histogram equalization and again merge the images
def histogramEqualization(image):
    #splits the image in bgr channels so that HE can be performed on each channel 
    b , g , r = cv2.split(image)
    b_h = channel(b)
    g_h = channel(g)
    r_h = channel(r)

    final_image = cv2.merge((b_h , g_h , r_h))
    return final_image

def AHE(image):
    #divide the image into 8x8 grid
    #good to go
    
    height = image.shape[0]
    width = image.shape[1]
    
    final = np.empty_like(image,dtype = np.uint8)
    width_ = int(width/8)
    height_ = int(height/2)
    for i in range(0, width , width_):
        for j in range(0,height , height_):
            box = image[j:j+height_ , i:i+width_,:]
            boxes = histogramEqualization(box)
 
            final[j:j+height_ , i:i+width_ , : ] = boxes

    return final

count = 1
if method == '1':
    print('Doing Histogram Equalization')
    counter = 1
    
    for image in images:
        print('Shape of image for method 1' ,image.shape)
        image1 = histogramEqualization(image)
        print(image.shape)
        plt.imshow(image1)
        plt.axis('off')
        plt.title('Histgram Equalization for image ' + str(counter))
        # plt.imsave("/Users/sheriarty/Desktop/enpm673/enpm673Proj2/Outputs/HE/HE_Image" + str(counter) + '.png' , image1)
        counter += 1
        plt.show()

if method == '2':
    print('Doing Adaptive Histogram Equalization')
    counter = 1
    for image in images:
        print('Shape of original image' , image.shape)

        images = cv2.resize(image , (1224,368)  ,interpolation = cv2.INTER_LINEAR)
        print('shape of image' , images.shape)
        final_image = AHE(images)
        plt.imshow(final_image)
        plt.axis('off')
        plt.title('Adaptive Histgram Equalization for image ' + str(counter))
        # plt.imsave("/Users/sheriarty/Desktop/enpm673/enpm673Proj2/Outputs/AHE/AHE_Image" + str(counter)+ '.png' , final_image)
        counter += 1
        plt.show()









    
    




