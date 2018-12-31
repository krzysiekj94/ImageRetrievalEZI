import math
from IPython.display import display, HTML
from skimage import data, io, filters, exposure
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters import gabor_kernel
from skimage.measure import moments, moments_central, moments_normalized, moments_hu
import numpy as np
from skimage.transform import resize
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass
from scipy import spatial
from pylab import *
import random    


class LabImage:
    def __init__(self, img, ID):
        self.img = img
        self.ID = str(ID)
        self.featuresValues = []


class Exercise:
    def __init__(self, folderPath, featuresNames = None, featuresExtractor = None):
        self.images = []
        self.featuresNames = []
        self.featuresExtractor = featuresExtractor
        self.loadAndCompute(folderPath, featuresNames, featuresExtractor)
        self.folderPath = folderPath
        
    def loadAndCompute(self, folderPath, featuresNames = None, featuresExtractor = None):
        collection = io.imread_collection(folderPath + "*.jpg")
        self.images = [LabImage(collection[i], collection.files[i]) for i in range(0, len(collection))]
        if featuresNames is not None:
            self.featuresNames = featuresNames
        if featuresExtractor is not None:
            for image in self.images:
                image.featuresValues = featuresExtractor(image.img)
                
    def searchForImage(self, folderPath, fileName, functionOfSImilarity, scale = 1.0,):
        queryImage = LabImage(io.imread(folderPath + fileName), fileName)
        queryImage.featuresValues = self.featuresExtractor(queryImage.img)
        orderedIDs = [i for i in range(0, len(self.images))]
        similarities = [functionOfSImilarity(queryImage.featuresValues, image.featuresValues) 
                        for image in self.images]
        
        tmp = list(zip(orderedIDs, similarities))
        tmp.sort(key = lambda x: x[1], reverse = True)
        orderedIDs, similarities = zip(*tmp)
        ### DISPLAY QUERY
        figure(figsize=(5.0 * scale, 5.0 * scale), dpi=80)
        subplot(1, 1, 1); plt.imshow(queryImage.img)
        plt.show()
        print(queryImage.ID)
        #features 
        data = []
        for j in range(0, len(self.featuresNames)):
            if isinstance(queryImage.featuresValues, np.ndarray): 
                fV = []
                fV.append("- "+ self.featuresNames[j])
                fV.append("NaN")
                data.append(fV)
            else:
                fV = []
                fV.append("- "+ self.featuresNames[j])
                fV.append("{:10.2f}".format(queryImage.featuresValues[j]))
                data.append(fV)
        
        col_width = max(len(word) for row in data for word in row) + 5 # PADDING
        for row in data:
            print("".join(word.ljust(col_width) for word in row))
        ### DISPLAY RESULTS
        self.display(scale = scale, orderedIDs = orderedIDs, similarities = similarities)
    
    def printStats(self, lIMG, rIMG, orderedIDs = [], similarities = []):
        data = []
        # names
        names = []
        for i in range(lIMG, rIMG):
            names.append(self.images[orderedIDs[i]].ID[len(self.folderPath):])
            names.append(" ")
        data.append(names)
        # similarities
        if len(similarities) > 0:
            sim = []
            for i in range(lIMG, rIMG):
                sim.append("Similarity")
                sim.append("{:10.2f}".format(similarities[i]))
            data.append(sim)
        #features 
        for j in range(0, len(self.featuresNames)):
            if isinstance(self.images[orderedIDs[i]].featuresValues, np.ndarray):
                fV = []
                for i in range(lIMG, rIMG):
                    fV.append("- "+ self.featuresNames[j])
                    fV.append("NaN")
                data.append(fV)         
            else:
                fV = []
                for i in range(lIMG, rIMG):
                    fV.append("- "+ self.featuresNames[j])
                    fV.append("{:10.2f}".format(self.images[orderedIDs[i]].featuresValues[j]))
                data.append(fV)
        
        col_width = max(len(word) for row in data for word in row) + 5 # PADDING
        for row in data:
            print("".join(word.ljust(col_width) for word in row))
            
    def display(self, scale = 1.0, orderedIDs = [], similarities = []):
        div = 3
        h = (len(self.images))/div
        if len(orderedIDs) == 0:
            orderedIDs = [i for i in range(0, len(self.images))]
        for i in range(0, math.ceil(h)):
            figure(figsize=(14.0 * scale, 5.0 * scale), dpi=80)
            idx = i * div
            for j in range(0, div):
                if (idx + j < len(self.images)):
                    subplot(2, div, j + 1); plt.imshow(self.images[orderedIDs[idx + j]].img)
            plt.show()
            end = min([len(self.images), i * div + div])
            self.printStats(i * div, end, orderedIDs = orderedIDs, similarities = similarities)

featuresNames_MeanColors = ["red mean", "green mean", "blue mean"]
featuresNames_StdColors =  ["red std", "green std", "blue std"]
featuresNames_SkewColors = ["red skew", "green skew", "blue skew"]

def featuresExtractor_MeanColors(image):
    values = [np.mean(image[:,:,0]),0,0]
    return values

def featuresExtractor_StdColors(image):
    values = [np.std(image[:,:,0]),0,0]
    return values

def featuresExtractor_SkewColors(image):
    values = [skew(image[:,:,0].ravel()), 0, 0 ]
    return values

def cosineSimilarity( featuresQuery, featuresImage ):
    val = 1.0 - spatial.distance.cosine( np.array( featuresImage ), np.array( featuresQuery ))
    return val

def euclideanSimilarity(featuresQuery, featuresImage):
    val = 1.0 - spatial.distance.euclidean( np.array( featuresImage ), np.array( featuresQuery ))
    return val

def exercise1():
    print("Exercise 1")

    #print("\tExercise 1.1")
    #e1 = Exercise("images/exercise1/intro/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    #e1.display(scale = 1.0)

    #print("\tExercise 1.2")
    #e1 = Exercise("images/exercise1/intro/", featuresNames_StdColors, featuresExtractor_StdColors)
    #e1.display(scale = 1.0)

    #print("\tExercise 1.3")
    #e1 = Exercise("images/exercise1/intro/", featuresNames_SkewColors, featuresExtractor_SkewColors)
    #e1.display(scale = 1.0)

    #print("\tExercise 1.4")
    #e1 = Exercise("images/exercise1/base/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    #e1.display(scale = 1.0)

    #print("\tExercise 1.5")
    e1 = Exercise("images/exercise1/base/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    #e1.searchForImage("images/exercise1/query/", "q2_green.jpg", cosineSimilarity )

    #print("\tExercise 1.6")
    #e1.searchForImage("images/exercise1/query/", "q2_green.jpg", cosineSimilarity)

    #print("\tExercise 1.7")
    #e1.searchForImage("images/exercise1/query/", "q3_blue.jpg", cosineSimilarity)

    #print("\tExercise 1.8")
    #e1.searchForImage("images/exercise1/query/", "q4_white.jpg", cosineSimilarity)

    #print("\tExercise 1.9")
    #e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", cosineSimilarity)

    #print("\tExercise 1.10")
    #e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", euclideanSimilarity)

    #print("\tExercise 1.11")
    #e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", euclideanSimilarity)

    print("\tExercise 1.12")
    e1.searchForImage("images/exercise1/query/", "q4_white.jpg", euclideanSimilarity)

#centroid
featuresNames_CentroidColors = ["red x", "red y",
                 "green x", "green y",
                 "blue x", "blue y"]
featuresNames_Hu = ["hu1", "hu2", "hu3","hu4", "hu5", "hu6","hu7"]

def featuresExtractor_CentroidColors( image ):
    valueList = []

    for x in range(0, 6):
        valueList.append( center_of_mass( image[x] ) )
    valueList = [i[0] for i in valueList]
        
    return valueList

norm = lambda x: -np.sign(x)*np.log10(np.abs(x))


def featuresExtractor_Hu(image):
    img = rgb2gray(image)
    hu = moments_central(img)
    hu = moments_central(hu)
    hu = moments_hu(hu)
    l = [norm(f) for f in hu]
    return l

def exercise2():
    print("Exercise 2")
    print("\tExercise 2.1")
    e2 = Exercise("images/exercise1/intro/", featuresNames_CentroidColors, featuresExtractor_CentroidColors)
    e2.display(scale = 1.0)

    print("\tExercise 2.2a).")
    scale = 1.0
    figure(figsize=(9.0 * scale, 4.0 * scale), dpi=80)
    subplot(2, 2, 1); plt.imshow(io.imread("images/exercise2/base/hu1.jpg")) 
    subplot(2, 2, 2); plt.imshow(io.imread("images/exercise2/base/hu2.jpg")) 
    subplot(2, 2, 3); plt.imshow(io.imread("images/exercise2/base/hu3.jpg")) 
    subplot(2, 2, 4); plt.imshow(io.imread("images/exercise2/base/hu4.jpg")) 
    plt.show()

    print("\tExercise 2.2b).")
    e2 = Exercise("images/exercise2/base/", featuresNames_Hu, featuresExtractor_Hu)

    print("\tExercise 2.3")
    e2.searchForImage("images/exercise2/query/", "q1.jpg", cosineSimilarity)

    print("\tExercise 2.4")
    e2.searchForImage("images/exercise2/query/", "q2.jpg", cosineSimilarity)

def exercise3():
    

#===== EXECUTE MAIN ======

def main():
    #exercise1()
    #exercise2()
    exercise3()

main()
