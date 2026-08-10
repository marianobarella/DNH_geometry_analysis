# DNH_geometry_analysis
Automation of Double Nanohole (DNH) SEM image analysis. Retrieval of DNH geometry from SEM images.

The code runs in 3 steps:
1st step) Crop and thresholding. 
2nd step) Find parameters from the binary image create in step 1.
3rd step) Do stats from the analyzed images, in batch.

The code has several outputs. Each step creates a folder where you can find the cropped images, the thresholded ones, and also the smoothed binary ones from which the contour of the DNH is retrieved. Later, the step two fits that contour and creates a CSV file with a list of retrieved paramters for each analyzed image.

It is recommended to run step 1 a couple of times on test images to find the right cropping size and to check if the thresholding method is working for your images.
