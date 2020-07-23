import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob



img_data = []   # an empty list to store image matrix
img_list = []   # an empty list to store image address 
img_list1 = [] 
R_images = []

# to get the addresses of all images
for i in range(1,16):    
    path = 'D:\Data_Science_Project\Term project 2020\Dataset_Question1' + '\\' + str(i) 
    file_names = glob.glob(path + '/*.pgm')
    img_list.append(file_names)

for i in range(15):
    img_list1 += img_list[i]

#to read all images
for i in img_list1:
    image = img.imread(i)
    img_data.append(image)


U = np.zeros((64,1))
VT = np.zeros((1,64))
s = np.zeros((2))

#number of singular values to be considered
num = 15

count = 0
for i in range(0,150):
  #Singular-value decomposition
    Udash, sdash, VTdash = np.linalg.svd(img.imread(img_list1[i]))
    if count < 10:
        U = np.hstack((U,Udash[:,:num]))
        s = np.hstack((s,sdash[:num]))
        VT = np.vstack((VT,VTdash[:num,:]))
    count = count + 1
    if count ==10:
        U = np.delete(U,0,axis=1)
        VT = np.delete(VT,0,axis=0)
        s = np.delete(s,[0,1])
        R = np.dot(U,np.dot(np.diag(s),VT))
        R = R/10
        R_images.append(R)
        U = np.zeros((64,1))
        VT = np.zeros((1,64))
        s = np.zeros((2))
        count = 0

# to define mean square error
def mse(imageA,imageB):
    err = np.sum((imageA.astype('float') - imageB.astype('float'))**2)
    err /= 4096
    return err

# to define difference between 
def norm(imageA,imageB):
    diff = np.linalg.norm(imageA - imageB)
    return diff

count1 = 0
test = []


for i in range(150):
    n = []
    for j in range(15):
        diff = norm(R_images[j],img.imread(img_list1[i]))
        n.append(diff)
    
    result = np.where(n==np.amin(n))
    test.append(result[0])
    print(result[0])

