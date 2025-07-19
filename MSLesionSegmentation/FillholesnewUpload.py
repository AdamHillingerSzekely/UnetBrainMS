import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, label, find_objects
direc = 'Training_Labels/'
numberoffiles = len(os.listdir(direc))
print(numberoffiles)
data_dir = 'NewDatasets/Training_Labels/Q1'

	# Load the binary NIfTI file
for filename in os.listdir(direc):
	img = os.path.join(direc, filename)	
	img = nib.load(img)
	data = img.get_fdata()

	# Threshold the image
	data_thresh = data < 1  # Holes are represented by 0s

	# Use connected component labeling to identify the holes
	labels, num_labels = label(data_thresh)

	# Calculate the volume of each hole
	for i in range(1, num_labels + 1):
	    mask = labels == i
	    volume = np.sum(mask)
	    if volume > 13.5:
	        # Fill the hole using a binary closing operation
	        struct = np.ones((5, 5, 5))  # Use a 5x5x5 structuring element
	        mask_closed = binary_closing(mask, structure=struct)
	        data[mask_closed] = 1

	# Save the filled image as a new binary NIfTI file
	img_filled = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
	nib.save(img_filled, os.path.join(data_dir, filename))