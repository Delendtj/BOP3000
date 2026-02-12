
My First Project - v3 Updated dataset M/F
==============================

This dataset was exported via roboflow.com on January 23, 2026 at 12:41 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1052 images.
Objects are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -10 and +10 degrees
* Random shear of between -12° to +12° horizontally and -13° to +13° vertically
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 1.49 percent of pixels


