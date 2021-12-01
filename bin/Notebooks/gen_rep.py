import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.filters import gabor_kernel
from skimage import io
import cv2 as cv
import os
import csv



def organize_data(data, labels):
    d = data.copy()
    l = labels.copy()
    position = []
    for i in range(len(data)):
        temp = labels[i].split('=')
        if len(temp) == 1:
            position.append(4)
            d[4] = data[i]
            l[4] = labels[i]

        else:
            loc = int(temp[1])
            position.append(loc)
            d[loc] = data[i]
            l[loc] = labels[i]
    #print(position)
    #print(l)
    return d , l

def num_label(label):
    nl = -1
    temp = label.split('!')[0]
    #print(temp)
    temp = temp.split()
    health = temp[0]
    #print(health)
    inhib = temp[2]
    #print(inhib)
    if (health == "HC" and inhib == "DMSO"):
        nl=0
    elif (health == "HC" and inhib == "CHIR"):
        nl=1
    elif (health == "HC" and inhib == "CHIR+TRICI"):
        nl=2
    elif (health == "HC" and inhib == "SGK1"):
        nl=3
    elif (health == "HC" and inhib == "CHIR+SGK1"):
        nl=4
    elif (health == "SCZ" and inhib == "DMSO"):
        nl=5
    elif (health == "SCZ" and inhib == "CHIR"):
        nl=6
    elif (health == "SCZ" and inhib == "CHIR+TRICI"):
        nl=7
    elif (health == "SCZ" and inhib == "SGK1"):
        nl=8
    elif (health == "SCZ" and inhib == "CHIR+SGK1"):
        nl=9
    else:
        print("error finding the number label")
        print(label)
        print(health)
        print(inhib)
    return nl

def create_img_collection(subdir, path):
    images = []
    img_names = []
    filelist = os.listdir(subdir)
    for img_name in filelist:
        file_name = subdir+img_name
        temp_img = io.imread(file_name)
        temp_img = img_as_float(temp_img)
        temp_name = img_name.split('.t')[0]
        #print(temp_name)
        images.append(temp_img)
        label = path+"!"+temp_name
        img_names.append(label)
    return img_names, images

def build_mask( image ):
    #img = image.copy()
    shape=image.shape
    img = np.zeros((shape[0]+20, shape[1]+20, 3))
    img[10:shape[0]+10,10:shape[1]+10,:]=image.copy()
    #print(img.shape)
    #img = img[:, :, ::-1] #switch from RGB to BGR 
    img = img_as_ubyte(img)
    # Convert BGR to HSV
    #img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # define range of yellow color in HSV
    lower_yellow = np.array([20,95,95])
    upper_yellow = np.array([40,255,255])
    # Threshold the HSV image to get only yellow colors
    mask = cv.inRange(img_hsv, lower_yellow, upper_yellow)
    #plt.imshow(mask)
    #plt.show()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask.shape[:2]
    blank_mask = np.zeros((h+2, w+2), np.uint8)
    #plt.imshow(blank_mask,cmap='Greys')
    #plt.show()
    # Floodfill from point (0, 0)
    cv.floodFill(mask, blank_mask, (0,0), 255)
    #plt.imshow(mask,cmap='Greys')
    #plt.show()
    # Invert floodfilled image
    mask = cv.bitwise_not(mask)
    #print("just printed the mask")
    #plt.imshow(mask, cmap='Greys')
    #plt.show()
    return mask


def write_to_csv(filename, csv_array):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        for row in csv_array:
            writer.writerow(row)
            
def build_ind_soma(images, mask):
    soma=[]
    soma_mask=[]
    #print("The mask shape is")
    #print(mask.shape)
    #print("The number of images is")
    #print(len(images))
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    #print("The number of connected components is")
    #print(n_labels-1)
    #print("The shape of the labels")
    #print(labels.shape)
    #print("printed the labels image")
    #io.imshow(labels)
    #io.show()
    for label in range(1,n_labels):
        width = stats[label, cv.CC_STAT_WIDTH]
        height = stats[label, cv.CC_STAT_HEIGHT]
        x = stats[label, cv.CC_STAT_LEFT]
        y = stats[label, cv.CC_STAT_TOP]
        #print(width)
        #print(height)
        #print(x)
        #print(y)
        if (width > 8 and height > 8):
            roi = labels[y-5:y + height+5, x-5:x + width+5].copy() # create a copy of the interest region from the labeled image
            roi[ roi != label] = 0  # set the other labels to 0 to eliminate intersections with other labels
            roi[ roi == label] = 255 # set the interest region to white
            roi=roi.astype(np.uint8)
            #io.imshow(roi)
            #io.show()
            img_col=[]
            for image in images:
                num_chan = len(image.shape)
                if num_chan == 2:
                    shape=image.shape
                    img = np.zeros((shape[0]+20, shape[1]+20), dtype=np.float32)
                    img[10:shape[0]+10,10:shape[1]+10]=image
                    #print(img.shape)
                    #img = img_as_ubyte(img)
                    img=img[y-5:y + height+5, x-5:x + width+5].copy()
                    #io.imshow(img)
                    #io.show()
                    res = cv.bitwise_and(img, img, mask= roi)
                    #io.imshow(res)
                    #io.show()
                    img_col.append(res)
                else:
                    shape=image.shape
                    img = np.zeros((shape[0]+20, shape[1]+20, shape[2]))
                    img[10:shape[0]+10,10:shape[1]+10,:]=image
                    #print(img.shape)
                    img = img_as_ubyte(img)
                    img=img[y-5:y + height+5, x-5:x + width+5].copy()
                    #io.imshow(img)
                    #io.show()
                    res = cv.bitwise_and(img, img, mask= roi)
                    #io.imshow(res)
                    #io.show()
                    #res=img_as_float(res)
                    img_col.append(res)
            soma.append(img_col)
            soma_mask.append(roi)
        else:
            print("Check the connected component height and width")
            print(width)
            print(height)

    return soma, soma_mask

def check_dir(path):
    #print(path)
    temp_name = path.split()[2]
    #print(temp_name)
    ret = False
    #print(temp_name)
    if (temp_name == "DMSO" or temp_name == "CHIR"):
        ret = True
        #print(temp_name+"success")
    return ret 



file_dir = os.path.join(os.getcwd(), "..", "..", "Images")
listdir = os.listdir(file_dir)
write_path = os.path.join(os.getcwd(), "..", "..", "r")
save_path = os.path.join(os.getcwd(), "..", "..", "Rep_Imgs")
#write_name = write_path+"/"+"data.csv"
    

mask_collection = []
data_collection = []
data_labels = []
data_ilabels=[]
out_csv=[]
out_csv.append(['ID', 'IMAGE', 'MASK', 'ClassLABEL', 'ImageTYPE'])
for image_dir in listdir:
    #print(image_dir)
    im_fold = os.listdir(file_dir+"/"+image_dir)
    #print(im_fold)
    for path in im_fold:
        #print(path)
        if check_dir(path):
            subdir = file_dir+"/"+image_dir+"/"+path+"/"
            #filelist = os.listdir(subdir)
            labels, images = create_img_collection(subdir, path)
            #print(labels)
            #print(subdir)
            d, l = organize_data(images, labels)
            #data_collection.append(d)
            #data_labels.append(l)
            nl = num_label(l[0])
            #print(nl)
            mask = build_mask(d[4])
            #print(l[4])
            ind_imgs, ind_msk = build_ind_soma(d, mask)
            l = [image_dir+"-"+x for x in l]
            #print(l[4])
            for i in range(len(ind_imgs)):
                data_collection.append(ind_imgs[i])
                index=str(i)
                #print([l,index])
                data_labels.append([l,index])
                data_ilabels.append(nl)
                mask_collection.append(ind_msk[i])





for i in range(len(data_collection)):
    for j in range(len(data_collection[i])):
        #print(len(mask_collection[i]))
        label = data_labels[i]
        #print(label)
        #print(label[0][j])
        if (j!=4):            
            im_type = label[0][j].split('=')[1]
            temp = label[0][j].split('-')
            #print(label[0][j])
            #print(temp)
            info1 = temp[0]+"-"
            info2 = temp[1].split('MA')[0]
            info3 = temp[2]
            #print(im_type)
            temp_path = save_path+"/"+info1+info2+info3+"_"+label[1]
            im_path = temp_path+".tif"
            #print(im_path)
            m_path = temp_path+"_mask.tif"
            #print(m_path)
            #print(data_ilabels[i])
            csv_row=[info1+info2+info3+"_"+label[1], im_path, m_path, str(data_ilabels[i]), str(im_type)]
            #print(csv_row)
            out_csv.append(csv_row)
            #print(data_collection[i][j].shape)
            #print(mask_collection[i].shape)
            #io.imshow(data_collection[i][j])
            #io.show()
            #io.imshow(mask_collection[i])
            #io.show()
            io.imsave(im_path, data_collection[i][j])
            #io.imsave(m_path, mask_collection[i])
        else:
            im_path = save_path+"/"+label[0][j]+"_"+label[1]+".tif"
            #print(im_path)
            #io.imshow(data_collection[i][j])
            #io.show()
            io.imsave(im_path, data_collection[i][j])

#write_to_csv(write_name, out_csv)
