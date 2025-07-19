import numpy as np
import nibabel as nib
from numpy import savetxt
from skimage import measure, morphology
from scipy import ndimage
import os
counter=1
direc = 'Training_Labels/'
#numberoffiles = len(os.listdir(direc))
#print(numberoffiles)
# Create a file to write the information
output_file = open("hole_info.txt", "w")

for filename in os.listdir(direc):
	img = os.path.join(direc, filename)
	img = nib.load(img)
	# checking if it is a file
	# Convert image data to numpy array
	dataorig = img.get_fdata()
	# Fill small holes using binary opening - converts 0 values to false and True
	data = ndimage.binary_opening(dataorig)


	# Label connected components as integer values, assigning each 'island' of non-zero values an integer value, 
	#value assigned to each island is determined by going right to left 
	labels, num_labels = ndimage.label(data)
	#print(labels)
	# Calculate size of each hole in terms of voxel, num_labels indicate the combined value of the islands of the non-zero values from the data matrix
	sizes = ndimage.sum(data, labels, range(num_labels + 1))
	#print(sizes)


	# Label connected components again
	labels, num_labels = ndimage.label(data)

	# Calculate number of holes
	num_holes = num_labels - 1  # exclude background label

	# Calculate volume of each hole
	voxel_size = np.prod(img.header.get_zooms())  # calculate voxel size


	hole_volumes = ndimage.sum(data, labels, range(1, num_labels + 1)) * voxel_size

	print(filename)
	print(f"Number of holes: {num_holes}")
	for i in range(num_holes):
		print(f"Hole {i+1} volume: {hole_volumes[i]}")


	    # Write information to the file
	output_file.write(f"Filename: {filename}\n")
	output_file.write(f"Number of holes: {num_holes}\n")
	for i in range(num_holes):
		output_file.write(f"Hole {i+1} volume: {hole_volumes[i]}\n")
	output_file.write("\n")

# Close the file after writing
output_file.close()
