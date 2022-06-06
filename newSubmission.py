import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import math


print("Please enter Query image directory")
imgPath=input()# Query image Directory path
print("Please enter Target image directory")
shelfPath=input()#Target image Directory path
objectDict={}#Object feature dictionary
shelfDict={}#Shelf feature dictionary
df=pd.DataFrame()

siftDetector=cv2.SIFT_create(nfeatures=1000)#Initialize SIFT detector with 1000 features

FLANN_INDEX_KDTREE = 0
indexParameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)#Initialize FLANN matcher
searchParameters = dict(checks = 50)

flannMatcher = cv2.FlannBasedMatcher(indexParameters, searchParameters)

#Read all images and store features
for objects in os.listdir(imgPath):
    img=cv2.imread(imgPath+objects,cv2.COLOR_BGR2GRAY)#Read images in Grayscale
    imgExt,ext=objects.split('.')#get Image name
    imgNumber =''.join(filter(lambda i: i.isdigit(), imgExt))#Get image number
    objectKeyPts,objectDesc=siftDetector.detectAndCompute(img,None)#Detect keypoints and decriptors
    imgH,imgW=img.shape[:2]#Get image dimensions
    objectDict[imgNumber]=(objectKeyPts,objectDesc,imgH,imgW)#Store information in a key value pair
    
print("Finished reading {} object files".format(len(objectDict)))

for shelf in os.listdir(shelfPath):
    img=cv2.imread(shelfPath+shelf,cv2.COLOR_BGR2GRAY)#Read images in Grayscale
    imgExt,ext=shelf.split('.')#Get image name
    imgNumber =''.join(filter(lambda i: i.isdigit(), imgExt))# Get image number
    shelfKeyPts,shelfDesc=siftDetector.detectAndCompute(img,None)#Detect keypoints and decriptors
    shelfDict[imgNumber]=(shelfKeyPts,shelfDesc)#Store information to reduce information access time

print("Finished reading {} shelf files".format(len(shelfDict)))


for i in range(1, len(objectDict)+1):
    print('i-->{}'.format(i))
    objFeatures=objectDict['{}'.format(i)]#Load query image features from Dictionary
    objKeyPts=objFeatures[0]#Load keypoints for Image "i"
    objDescPts=objFeatures[1]#Load descriptors for Image "i"
    objH=objFeatures[2]#Load Image height
    objW=objFeatures[3]#Load Image width
    for j in range(1, len(shelfDict)+1):
        shlfFeatures=shelfDict['{}'.format(j)]#Load target image features from Dictionary
        shlfKeyPts=shlfFeatures[0]#Load keypoints for Image "j"
        shlfDescPts=shlfFeatures[1]#Load keypoints for Image "j"
        totalMatch=flannMatcher.knnMatch(objDescPts,shlfDescPts,k=2)#Match image decriptors with K-Nearest Neighbors
        goodPoints=[]#Initialize list to store good Points
        goodPoints.clear()
        for m,n in totalMatch:
            if m.distance< 0.74*n.distance:#Compute using Lowe's Ratio Test
                goodPoints.append(m)#Append if condition is satisfied
        if len(goodPoints)>20:
            objPts=np.float32([ objKeyPts[m.queryIdx].pt for m in goodPoints ]).reshape(-1,1,2)#Get Matching Points for Query image
            shlfPts=np.float32([ shlfKeyPts[m.trainIdx].pt for m in goodPoints ]).reshape(-1,1,2)#Get matching points for Target image
            matrix,mask=cv2.findHomography(objPts,shlfPts,cv2.RANSAC,5.0)#Compute Homography Matrix
            if matrix is None:
                print("Cannot compute Homography matrix")
                
            else:
                height=math.ceil(objH*.30)#Compute Bounding Box Height
                width=math.ceil(objW*.50)#Compute Bounding Box Width
                detectedEdges = np.float32([ [0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0] ]).reshape(-1, 1, 2)
                transformedCorners = abs(cv2.perspectiveTransform(detectedEdges, matrix))#Detect corner        
                x=int(transformedCorners[0][0][0])#get X_Min
                y=int(transformedCorners[0][0][1])#get Y_Min
                '''
                if (j%10==0):
                    print("Generating Test File")
                    img=cv2.imread(shelfPath+'db{}.jpg'.format(j))
                    dispImg=cv2.rectangle(img, (x,y), (x+width,y+height), (255,0,0), 1,cv2.LINE_AA)  //Uncomment if you want to generate localized bounding box for every match
                    plt.imshow(dispImg)
                    plt.draw()
                    plt.pause(3)
                    plt.close('all')
                    '''
                dataFromIteration={'Img':i, 'Target':j, 'XMin':x,'XMax':x+width,'YMin':y,'YMax':y+height}#Store bounding box coordinates in dictionary
                df=df.append(pd.Series(dataFromIteration),ignore_index=True)#Add dictionary to Pandas Dataframe
                dataFromIteration.clear()#Clear dictionary
        else:
            print("Insufficient Matches")
            

df.to_csv("D:/28_GitHub/Infilect Task/sandbox/Custom DataSet/solution_1.csv",sep=' ',index=False)#Save file as CSV
