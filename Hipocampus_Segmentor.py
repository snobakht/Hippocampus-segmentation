'''Samaneh Nobakht UCI: 10059083 --- Hipocampus segmentation -- Image processing course term paper
This script Takes the probability map produced by atlas based segmentation (MNI atlas and harvard-oxford subcortical structural probabilistic atlas), pre-process it, runs the
developed algorithm, post process it and returns the  segmented hipocampus with average 74% dice coeficiency based on 4 samples.

Algorithm in general follows these steps:

Step1-Register MNI atlas to patient space, use transformation to apply to probabilistic map of hipocampus from harvard-Oxford subcortical structural to segment approximate region
of patient hipocampus. Result is a probabilistic map of patient hipocampus.
Step2-Convert result from step1 to binary mask and Multiply mask by image to find forground and a zero Background.
Forground has a similar shape to hipocampus, but is contaminated by other area voxels, such as csf, amigdala, whiteMatter.
Step3-Crop step 2 image and probability map and the binary mask to as much as possible(for faster computation).
Step4-Use step 3 forground image to cluster it based on k mean clustering to similar intensity regions.
Step5-Use step 4 clustered image and for each one of clusters multiply it elementwise to the probability map- this makes sure to give certain weights to each pixels of each cluster.
Step6-For each result from step4 cluster it two 2 group based on k mean clustering but this time replace voxels of each cluster with the avarage values of the group.
Step7-Pick the cluster with highest avarge values.
Step8-Run islandremoval2D and dilateErode3D to remove small islands and to fill small holes respectively. Result is the segmented hipocampus.
'''

import vtk
import numpy
import math
import random
import argparse
import os
import sys
import scipy.optimize as optimize

#Create argument parser
parser = argparse.ArgumentParser(
    description = "A script that segments hipocampus")
#Add arguments to parser
parser.add_argument("input1",  help="Input should be a Nifti image which is a probability map of patient's hipocampus")
parser.add_argument("input2",  help="Input should be a Nifti image of patient's brain")
#Parse arguments
args = parser.parse_args()
if (not os.path.exists(args.input1)):
	print "ERROR: " + args.input1 + " does not exist!"
	sys.exit()

# Read the image data from a NIFTI file
reader1 = vtk.vtkImageReader()
ext = os.path.splitext(args.input1)[1]
if (ext == ".nii" or ext == ".nifti"):
		reader1 = vtk.vtkNIFTIImageReader()
		reader1.SetFileName(args.input1)
else:
		print "ERROR: image format not recognized for " + args.input1
		sys.exit()
reader1.Update()
probabilityMap=reader1.GetOutput()
if (not os.path.exists(args.input2)):
	print "ERROR: " + args.input2 + " does not exist!"
	sys.exit()

# Read the image data from a NIFTI file
reader2 = vtk.vtkImageReader()
ext = os.path.splitext(args.input2)[1]
if (ext == ".nii" or ext == ".nifti"):
		reader2 = vtk.vtkNIFTIImageReader()
		reader2.SetFileName(args.input2)
else:
		print "ERROR: image format not recognized for " + args.input2
		sys.exit()
reader2.Update()
OriginalImage=reader2.GetOutput()
'''Define all the required functions for segmentation'''
print("Defining required functions for segmentation...")
#......................................................
#A function that creates an image with same structure as input image but all voxels are zero
def ZeroImage (image):
    image1=vtk.vtkImageData()
    image1.DeepCopy(image)
    for z   in range(0,  image.GetDimensions()[2]):
     for y   in range(0,  image.GetDimensions()[1]):
       for x  in range (0,  image.GetDimensions()[0]):
        image1.SetScalarComponentFromFloat(x,y,z,0,0)
    return(image1)
#....................................................................
#A function that gathers all intensity values except zero in a list
def ExistingIntensities(image):
    IntensityValues=[]
    for z   in range(0,  image.GetDimensions()[2]):
      for y   in range(0,  image.GetDimensions()[1]):
        for x  in range (0,  image.GetDimensions()[0]):
            voxelValue= image.GetScalarComponentAsFloat(x,y,z,0)
            if voxelValue!=0:
             IntensityValues.append(voxelValue)
    return(IntensityValues)
#...........................................................
# A function that chooses as many desired numbers randomly from a list and returns them in an array
def RandomPicker (number, Alist):
    Array=numpy.zeros((number,1))
    for i in range(0,number):
        Array[i]=random.choice(Alist)
    return(Array)
#................................................................................
#A function to multiply binary mask by original image
def ForgroundImageFinder (BinaryMask, OriginalImage):
 ForgroundImage = vtk.vtkImageMathematics()
 ForgroundImage.SetOperationToMultiply()
 ForgroundImage.SetInput1Data(BinaryMask)
 ForgroundImage.SetInput2Data(OriginalImage)
 ForgroundImage.ReleaseDataFlagOff()
 writer=vtk.vtkNIFTIImageWriter()
 writer.SetInputConnection(ForgroundImage.GetOutputPort())
 writer.SetFileName("ForgroundImageout.nii")
 writer.Write()
 reader=vtk.vtkNIFTIImageReader()
 reader.SetFileName("ForgroundImageout.nii")
 reader.Update()
 image=reader.GetOutput()

 return(image)
#.................................................................................
# A function that crops image to desired VOI
# CroppedImage.SetVOI(48, 90,96,163, 99,137)

def ImageCropper(Image,X1, X2,Y1,Y2, Z1,Z2):
     CroppedImage = vtk.vtkExtractVOI()
     CroppedImage.SetInputData(Image)
     CroppedImage.SetVOI(X1, X2,Y1,Y2, Z1,Z2)
     CroppedImage.Update()
     writer=vtk.vtkNIFTIImageWriter()
     writer.SetInputConnection(CroppedImage.GetOutputPort())
     writer.SetFileName("CroppedImageout.nii")
     writer.Write()
     reader=vtk.vtkNIFTIImageReader()
     reader.SetFileName("CroppedImageout.nii")
     reader.Update()
     image=reader.GetOutput()
     return(image)
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#A function that finds extent for drawing a tight bounding box around a binary mask
def TightBounder(image):
   yvector=[]
   zvector=[]
   xvector=[]
   for z   in range(0,  image.GetDimensions()[2]):
     for y   in range(0,  image.GetDimensions()[1]):
       for x  in range (0,  image.GetDimensions()[0]):
         vox=image.GetScalarComponentAsFloat(x,y,z,0)
         if vox==1:
           yvector.append(y)
           zvector.append(z)
           xvector.append(x)
   xmin=min(xvector)
   xmax=max(xvector)
   zmin=min(zvector)
   zmax=max(zvector)
   ymin=min(yvector)
   ymax=max(yvector)
   Array=[xmin,xmax,ymin,ymax,zmin,zmax]
   return(Array)
#.................................................................................
#A function That converts a probability map to binary mask
def BinaryGenerator(ProbabilitymapImage):
    image=ZeroImage(ProbabilitymapImage)
    for z   in range(0,  ProbabilitymapImage.GetDimensions()[2]):
     for y   in range(0,  ProbabilitymapImage.GetDimensions()[1]):
       for x  in range (0,  ProbabilitymapImage.GetDimensions()[0]):
        vox= ProbabilitymapImage.GetScalarComponentAsFloat(x,y,z,0)
        if vox!=0:
          image.SetScalarComponentFromFloat(x,y,z,0,1)
        else:
            image.SetScalarComponentFromFloat(x,y,z,0,0)
    return(image)

#'''''''''''''''''''''''''''''''''''''''''''''''''''''
#A function that finds number of nonZero elements in a mask
def NonZeroFinder(BinaryMask):
 temp=0
 for z   in range(0,  Mask.GetDimensions()[2]):
     for y   in range(0,  Mask.GetDimensions()[1]):
      for x  in range (0,  Mask.GetDimensions()[0]):
          vox= Mask.GetScalarComponentAsFloat(x,y,z,0)
          if vox!=0:
              temp=temp+1
 return(temp)
#....................................................................
#A function that takes an image and creates other binary image with specific pixels of first image
def SpecificBinaryMaker (image, pixelvalue):
    Binaryimage=ZeroImage(image)
    for z   in range(0,  image.GetDimensions()[2]):
     for y   in range(0,  image.GetDimensions()[1]):
       for x  in range (0,  image.GetDimensions()[0]):
        vox= image.GetScalarComponentAsFloat(x,y,z,0)
        if vox==(pixelvalue):
            Binaryimage.SetScalarComponentFromFloat(x,y,z,0,1)
    return(Binaryimage)
#...........................................................................
#PostProcess function
def Post_Process(image):
     dilateErode=vtk.vtkImageDilateErode3D()
     dilateErode.SetInputData(image)
     dilateErode.SetDilateValue(1)
     dilateErode.SetErodeValue(0)
     dilateErode.SetKernelSize(3,3,1)
     dilateErode.ReleaseDataFlagOff()
     dilateErode.Update()
     dilateErode1=vtk.vtkImageDilateErode3D()
     dilateErode1.SetInputConnection(dilateErode.GetOutputPort())
     dilateErode1.SetErodeValue(1)
     dilateErode1.SetDilateValue(0)
     dilateErode1.SetKernelSize(3,3,1)
     dilateErode1.ReleaseDataFlagOff()
     dilateErode1.Update()
     return(dilateErode1)
#....................................................
# A function that multiplies kmean out put to probability mapper
def ElementwiseMultiplier (Binaryimage, probabilityMapImage):
    image2=ZeroImage(Binaryimage)
    for z   in range(0,  Binaryimage.GetDimensions()[2]):
     for y   in range(0, Binaryimage.GetDimensions()[1]):
       for x  in range (0,  Binaryimage.GetDimensions()[0]):
           voxel1= Binaryimage.GetScalarComponentAsFloat(x,y,z,0)
           voxel2=probabilityMapImage.GetScalarComponentAsFloat(x,y,z,0)
           image2.SetScalarComponentFromFloat(x,y,z,0,voxel1*voxel2)
    return(image2)
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#K mean clustering function that replaces each cluster with average of intensities of the cluster
def KmeanClustering(GreyScaleImage,numberOfClusters,numberOfIterations):
  ExistingIntensities(GreyScaleImage)
  UniqueIntensityArray=list(set(ExistingIntensities(GreyScaleImage)))
  ArrayOfCentroids=RandomPicker(numberOfClusters,UniqueIntensityArray)
  ClusteredImage = vtk.vtkImageData()
  ClusteredImage.DeepCopy(GreyScaleImage)
  for k in range(0,numberOfIterations):
   counter=numpy.ones((numberOfClusters,1))
   new_Cluster=numpy.zeros((numberOfClusters,1))
   for z   in range(0,  GreyScaleImage.GetDimensions()[2]):
    for y   in range(0,  GreyScaleImage.GetDimensions()[1]):
      for x  in range (0,  GreyScaleImage.GetDimensions()[0]):
       vox= GreyScaleImage.GetScalarComponentAsFloat(x,y,z,0)
       if vox!=0:
          dist=[]
          for i in range (0,numberOfClusters):
            LeastSQ=math.pow(vox-ArrayOfCentroids[i],2)
            dist.append(LeastSQ)
          for i in range(0,numberOfClusters):
             if min(dist)==dist[i]:
                counter[i]=counter[i]+1
                new_Cluster[i]=new_Cluster[i]+vox
                ClusteredImage.SetScalarComponentFromFloat(x,y,z,0,ArrayOfCentroids[i]+1)
   for i in range(0,numberOfClusters):
    ArrayOfCentroids[i]=new_Cluster[i]/counter[i]
  return(ClusteredImage)
#............................................................................

'''Call Functions to Segment Hipocampus'''
#Binarize probability map
image0=BinaryGenerator(probabilityMap)
print("Binarizing Probability Map done!")
#Find Region of Interest
image=ElementwiseMultiplier (image0, OriginalImage)
print("Region of Interest extracted!")

#Find tightest bounding box around mask
print("Extracting tightest bounding box around binary mask...")
Array=TightBounder(image0)
print("Image resizing in progress!")
#crop image using above bounding box
HipoCrop=ImageCropper(image,Array[0],Array[1],Array[2],Array[3],Array[4],Array[5])
writer1=vtk.vtkNIFTIImageWriter()
writer1.SetInputData(HipoCrop)
writer1.SetFileName("ROI_HipoCrop_"+args.input2)
writer1.Write()
#Crop probability map
probabilityMapCrop=ImageCropper(probabilityMap,Array[0],Array[1],Array[2],Array[3],Array[4],Array[5])
writer2=vtk.vtkNIFTIImageWriter()
writer2.SetInputData(probabilityMapCrop)
writer2.SetFileName("ProbMapCrop_Hipo_"+args.input2)
writer2.Write()
print("Image resizing done!")

#Set Number of Clusters, iterations
numberOfClusters=3
numberOfIterations=30
#Run kmeans to cluster image
print("K mean clustering running, this may take a few minutes!")
ClusteredImage=KmeanClustering(HipoCrop,numberOfClusters,numberOfIterations)
writer3=vtk.vtkNIFTIImageWriter()
writer3.SetInputData(ClusteredImage)
writer3.SetFileName("Clustered_Kmean_Hipo_"+args.input2)
writer3.Write()
print("Region of Interest clustered to three groups.")
#Find unique intensities in clustered image
UniqueIntensityArray=list(set(ExistingIntensities(ClusteredImage)))
# Set number of clusters for secondary k mean clustering
numberofsecondaryclusters=2
#Create an empty list to gather high intensity clusters id after secondary clustering
ClusterID=[]
#For each group in clustered image multiply elementwise by probability map  to give weight to each voxel in each cluster
print("Weighting clusters in progress!")
for j in UniqueIntensityArray:
    Binaryimage=SpecificBinaryMaker(ClusteredImage, j)
    WeightedImage=ElementwiseMultiplier(Binaryimage, probabilityMapCrop)
#Cluster weighted image two 2 clusters for second time
    ClusteredWeightedImage=KmeanClustering(WeightedImage,numberofsecondaryclusters,numberOfIterations)
#Gather intensities
    UniqueIntensityArray2=list(set(ExistingIntensities(ClusteredWeightedImage)))
    ClusterID.append(max(UniqueIntensityArray2))
#After secondary clustering choose cluster with highest weight or highest probability
index = ClusterID.index(max(ClusterID))
Binaryimage=SpecificBinaryMaker(ClusteredImage, UniqueIntensityArray[index])
WeightedImage=ElementwiseMultiplier(Binaryimage, probabilityMapCrop)
writer2=vtk.vtkNIFTIImageWriter()
writer2.SetInputData(WeightedImage)
writer2.SetFileName("WeightedCluster_Hipo_"+args.input2)
writer2.Write()
print("K mean clustering running on weighted clusters, this may take a few minutes!")
ClusteredWeightedImage=KmeanClustering(WeightedImage,numberofsecondaryclusters,numberOfIterations)
writer4=vtk.vtkNIFTIImageWriter()
writer4.SetInputData(ClusteredWeightedImage)
writer4.SetFileName("WeightedClusterClustered_Hipo_"+args.input2)
writer4.Write()
UniqueIntensityArray3=list(set(ExistingIntensities(ClusteredWeightedImage)))
print("Highest probability region extracted.")
SegmentedImage=SpecificBinaryMaker(ClusteredWeightedImage,max(UniqueIntensityArray3))
#Post process the segmented image
PostprocessOutput=Post_Process(SegmentedImage)
print("Segmented Hipocampus is written to memory.")
#Write segmented image
writer5=vtk.vtkNIFTIImageWriter()
writer5.SetInputConnection(PostprocessOutput.GetOutputPort())
writer5.SetFileName("Segmented_Hipo_"+args.input2)
writer5.Write()
writer6=vtk.vtkNIFTIImageWriter()
writer6.SetInputData(SegmentedImage)
writer6.SetFileName("SegmentedNoPostProcess_Hipo_"+args.input2)
writer6.Write()

#'''''''''''''''''''''''''''''''''''''''''''''''''''
#View segmented image
zslice=1
#Define viewer and set window and level and pass the segmented image
viewer = vtk.vtkImageViewer()
viewer.SetInputConnection(PostprocessOutput.GetOutputPort())
viewer.SetColorWindow(1000)
viewer.SetColorLevel(0)
#Define interactor and pass it to viewer
interact = vtk.vtkRenderWindowInteractor()
viewer.SetupInteractor(interact)
#Define function for scrolling through slides
def SlideChange(key,Event):
    global zslice
    #Get key symbol
    key=interact.GetKeySym()
     # If UP arrow was pressed slide will increase
    if key=="Up":
        zslice = zslice + 1
        viewer.SetZSlice(zslice)
        viewer.Render()
     # If Down arrow was pressed slide will increase
    if key=="Down":
        zslice = zslice - 1
        viewer.SetZSlice(zslice)
        viewer.Render()
# Set the observer atrribute to interactor passing the function slidechange to interact with user
interact.AddObserver("KeyPressEvent",SlideChange,1.0)
#Render viewer
viewer.Render()
#Start interactor
interact.Start()
