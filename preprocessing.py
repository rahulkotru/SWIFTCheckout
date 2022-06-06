import cv2
import os


# Static Method to convert RGB image to Grayscale
def toGray(img, name, grayPath):
    image = cv2.imread(img)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(grayPath+'{}.jpg'.format(name), grayImage)


#Static method to split filename
def removeExtension(filePath):
    name=os.path.splitext(os.path.basename(filePath))
    return name, name[0]
    
#Static method to resize given image    
def resize(img, name, dim, resizePath):
    image=cv2.imread(img)
    resizeImage=cv2.resize(image,(dim,dim))
    cv2.imwrite(resizePath+'{}.jpg'.format(name),resizeImage)

#Create List to store file names
objectList=[]
shelfList=[]

#Define Directory Structure
objectPath = r'../data/object/'
shelfPath = r'../data/shelf/'
resizeObject = r'../data/resizeObject/'
resizeShelf = r'../data/resizeShelf/'
grayObject = r'../data/grayObject/'
grayShelf = r'../data/grayShelf/'
resizeGrayObject = r'../data/resizeGrayObject/'
resizeGrayShelf = r'../data/resizeGrayShelf/'
#Create Directories
if not(os.path.exists(grayObject)):
    os.makedirs(grayObject)
if not(os.path.exists(grayShelf)):
    os.makedirs(grayShelf)
if not(os.path.exists(resizeGrayObject)):
    os.makedirs(resizeGrayObject)
if not(os.path.exists(resizeGrayShelf)):
    os.makedirs(resizeGrayShelf)
if not(os.path.exists(resizeObject)):
    os.makedirs(resizeObject)
if not(os.path.exists(resizeShelf)):
    os.makedirs(resizeShelf)

#Iterate over query Images
for path in os.listdir(objectPath):
    if os.path.isfile(os.path.join(objectPath, path)): 
        objectName,objectNameWithoutExtension=removeExtension(path)
        objectList.append(objectNameWithoutExtension)
        toGray((os.path.join(objectPath, path)), objectNameWithoutExtension, grayObject)
        resize((os.path.join(grayObject, path)), objectNameWithoutExtension, 640, resizeGrayObject)
        resize((os.path.join(objectPath, path)), objectNameWithoutExtension, 640, resizeObject)


        

#Iterate over target images
for path in os.listdir(shelfPath):
    if os.path.isfile(os.path.join(shelfPath, path)):
        shelfName, shelfNameWithoutExtension=removeExtension(path)
        shelfList.append(shelfNameWithoutExtension)
        toGray((os.path.join(shelfPath, path)), shelfNameWithoutExtension, grayShelf)
        resize((os.path.join(grayShelf, path)), shelfNameWithoutExtension, 640, resizeGrayShelf)
        resize((os.path.join(shelfPath, path)), shelfNameWithoutExtension, 640, resizeShelf)



