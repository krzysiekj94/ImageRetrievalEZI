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

    print("\tExercise 1.1")
    e1 = Exercise("images/exercise1/intro/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    e1.display(scale = 1.0)

    print("\tExercise 1.2")
    e1 = Exercise("images/exercise1/intro/", featuresNames_StdColors, featuresExtractor_StdColors)
    e1.display(scale = 1.0)

    print("\tExercise 1.3")
    e1 = Exercise("images/exercise1/intro/", featuresNames_SkewColors, featuresExtractor_SkewColors)
    e1.display(scale = 1.0)

    print("\tExercise 1.4")
    e1 = Exercise("images/exercise1/base/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    e1.display(scale = 1.0)

    print("\tExercise 1.5")
    e1 = Exercise("images/exercise1/base/", featuresNames_MeanColors, featuresExtractor_MeanColors)
    e1.searchForImage("images/exercise1/query/", "q2_green.jpg", cosineSimilarity )

    print("\tExercise 1.6")
    e1.searchForImage("images/exercise1/query/", "q2_green.jpg", cosineSimilarity)

    print("\tExercise 1.7")
    e1.searchForImage("images/exercise1/query/", "q3_blue.jpg", cosineSimilarity)

    print("\tExercise 1.8")
    e1.searchForImage("images/exercise1/query/", "q4_white.jpg", cosineSimilarity)

    print("\tExercise 1.9")
    e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", cosineSimilarity)

    print("\tExercise 1.10")
    e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", euclideanSimilarity)

    print("\tExercise 1.11")
    e1.searchForImage("images/exercise1/query/", "q5_dark.jpg", euclideanSimilarity)

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

#VARIABLE FOR EXERCISE 3
angles = [0, 30, 60, 90]
freqs = [0.03, 0.05, 0.07, 0.09]
scale = 1.0
kernels = []
std = 10.0

def applyGaborsFilters(image):
    return generateOutputImageUseGaborFilter(image)

def runBaseGaborFilterScript():
    #angles = [0, 45, 10, 65] #my example
    #freqs = [0.2, 0.3, 0.6, 0.9] #my example
    for angle in angles:
        kernels_row = []
        #figure(figsize=(14.0 * scale, 4.0 * scale), dpi=80)
        num = 0
        for freq in freqs:
            num += 1
            kernel = np.real(gabor_kernel(freq, theta=angle / 90.0 * 0.5 * np.pi, sigma_x=std, sigma_y=std))
            kernels_row.append(kernel)
            #subplot(1, 4, num); plt.imshow(kernel, cmap='jet', vmin=-0.002, vmax=0.002) 
            #plt.colorbar(orientation='horizontal', ticks = [-0.002, 0.0, 0.002])
        kernels.append(kernels_row)
    #plt.show()

def generateOutputImageUseGaborFilter(image):
    runBaseGaborFilterScript()
    
    if image is None:
        image = rgb2gray(io.imread("images/exercise3/base/b1.jpg"))
    else:
        image = rgb2gray(image)

    #io.imshow(image)

    # TODO: init sum_image with zeros (np. zero). The matrix must be of a proper size (image.shape)
    # DONE!
    sum_image = np.zeros(image.shape)
    
    for row in kernels:
        #figure(figsize=(14.0 * scale, 4.0 * scale), dpi=80)
        num = 0
        for kernel in row:
            num += 1
            img_convolve = fftconvolve(image, kernel, mode='same')  
        
            # TODO
            # add img_convovle to sum_image
            # DONE!
            np.add(sum_image,img_convolve)
            
            #subplot(1, 4, num); plt.imshow(img_convolve, cmap='jet', vmin=0.0, vmax=0.5) 
            #plt.colorbar(orientation='horizontal', ticks=[0.0, 0.5])
            
    # TODO compute the averaged values (divide sum_image by the number of kernels = 16)
    # DONE!
    averaged_image = np.divide(sum_image, 16.0)

    #plt.show()
    #figure(figsize=(4.0 * scale, 4.0 * scale), dpi=80)
    #subplot(1, 1, 1); plt.imshow(averaged_image, cmap='jet', vmin=0.0, vmax=0.5) 
    #plt.colorbar(orientation='horizontal', ticks=[0.0, 0.25, 0.5])
    #plt.show()

    return averaged_image


def displayGabors():
    collection1 = io.imread_collection("images/exercise3/base/*.jpg") 
    collection2 = io.imread_collection("images/exercise3/query/*.jpg")  
    images1 = [collection1[i] for i in range(0, len(collection1))]
    images2 = [collection2[i] for i in range(0, len(collection2))]
    images = images1 + images2
    for image in images:
        figure(figsize=(10.0 * scale, 5.0 * scale), dpi=80)
        gabor_image = applyGaborsFilters(image)
        subplot(1, 2, 1);
        plt.imshow(image) 
        subplot(1, 2, 2);
        plt.imshow(gabor_image, cmap='jet', vmin=0.0, vmax=0.5)
        plt.show()

featuresNames_Texture = ["Mean", "Std", "Skew","Kurt"]

def featuresExtractor_Texture(image):
    featuresValues = []
    img = applyGaborsFilters(image)
    
    mean = np.mean(image)
    std = np.std(image)
    skewValue = skew(image, axis=None)
    kurt = kurtosis(image, axis=None)

    featuresValues.append(mean)
    featuresValues.append(std)
    featuresValues.append(skewValue)
    featuresValues.append(kurt)
    #print(featuresValues)
    
    return featuresValues
 
def weightedSumSimilarity(featuresQuery, featuresImage):
    w = [1.0, 1.0, 1.0, 1.0]

    for x in range(0, len(w)):
        w[x] = abs(featuresQuery[x] - featuresImage[x])

    score = max(w)

    return score

def exercise3():
    print("Exercise 3")
    print("\tExercise 3.1, 3.2") 
    generateOutputImageUseGaborFilter(None)
    print("\tExercise 3.3 3.4")
    displayGabors()
    print("\tExercise 3.5")
    e3 = Exercise("images/exercise3/base/", featuresNames_Texture, featuresExtractor_Texture)
    e3.searchForImage("images/exercise3/query/", "q1.jpg", weightedSumSimilarity)
    e3.searchForImage("images/exercise3/query/", "q2.jpg", weightedSumSimilarity)
    e3.searchForImage("images/exercise3/query/", "q3.jpg", weightedSumSimilarity)
    e3.searchForImage("images/exercise3/query/", "q4.jpg", weightedSumSimilarity)
    
#===== EXECUTE MAIN ======

def main():
    exercise1()
    exercise2()
    exercise3()

main()
