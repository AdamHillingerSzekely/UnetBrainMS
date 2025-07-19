import os
from glob import glob
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Rotate90d,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandGaussianNoised,
    RandAffined,
    RandRotated,
    Activations,
    Flipd,
)
from first import first
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
data_dir = 'Niftis_complete'
'''
specific_data_rep = 'Niftis_complete/rotate90/Training_Images_Flair/Images/patient_generated_0.nii.gz'
orig = 'Niftis_complete/Training_Images_Flair_Copy/Patient1.nii.gz'

print('adapted nifti')
n2_img = nib.load(specific_data_rep)

print(n2_img)

print('original nifti')
n1 = nib.load(orig)
print(n1)
'''
print('hello')
train_images = sorted(glob(os.path.join(data_dir, 'First_Image', '*.nii.gz')))
train_labels = sorted(glob(os.path.join(data_dir, 'First_Image_Label', '*.nii.gz')))
val_images = sorted(glob(os.path.join(data_dir, 'Testing_Images_DP', '*.nii.gz')))
val_labels = sorted(glob(os.path.join(data_dir, 'Testing_Labels', '*.nii.gz')))

train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
#train_files = train_files[25:37]
#val_files = val_files[25:37]

pixdim=(1.5, 1.5, 1.0)
a_min=-200
a_max=200
spatial_size=[128,128,64]
generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,), 
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=["image", "label"]),
    ]
)
train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"), 
            CropForegroundd(keys=["image", "label"], source_key="image"),   
            ToTensord(keys=["image", "label"]),

        ]
    )

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        Resized(keys=["image", "label"], spatial_size=[128,128,64]),   
        ToTensord(keys=["image", "label"]),
    ]
)

generat_transforms2 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            #Orientationd(keys=["image", "label"], axcodes="RPI"),
            #Flipd(keys=["image", "label"], spatial_axis=1 )
            Rotate90d(keys=["image", "label"], k=1, spatial_axes=(1, 2)),
            #CropForegroundd(keys=["image", "label"], source_key="image"),
        	#Resized(keys=["image", "label"], spatial_size=[128,128,128]),    
            #ToTensord(keys=["image", "label"]),
        ]
    )

generat_transforms3 = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

'''
orig_ds = Dataset(data=val_files, transform=generat_transforms2)
orig_loader = DataLoader(orig_ds, batch_size=1)
orig_patient = first(orig_loader)

test_ds = Dataset(data=val_files, transform=generat_transforms3)
test_loader = DataLoader(test_ds, batch_size=1)
test_patient = first(test_loader)

print((test_patient["image"][0, 0, :, :, 1].shape))

for i in range(30, 35, 1):
    plt.figure("display", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Original patient slice {i}")
    plt.imshow(orig_patient["image"][0, 0, :, :, i], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"Generated patient slice {i}")
    plt.imshow(test_patient["image"][0, 0, :, :, i], cmap="gray")
    plt.show()
'''

def save_nifti(in_image, in_label, out, index):
    # Convert the torch tensors into numpy array
    volume = np.array(in_image.detach().cpu()[0, :, :, :], dtype=np.float32)
    lab = np.array(in_label.detach().cpu()[0, :, :, :], dtype=np.float32)
    
    # Convert the numpy array into nifti file
    volume = nib.Nifti1Image(volume, np.eye(4))
    lab = nib.Nifti1Image(lab, np.eye(4))
    
    # Create the path to save the images and labels
    path_out_images = os.path.join(out, 'Images')
    path_out_labels = os.path.join(out, 'Labels')
    
    # Make directory if not existing
    if not os.path.exists(path_out_images):
        os.mkdir(path_out_images)
    if not os.path.exists(path_out_labels):
        os.mkdir(path_out_labels)
    
    path_data = os.path.join(out, 'Images')
    path_label = os.path.join(out, 'Labels')
    nib.save(volume, os.path.join(path_data, f'Patient{index}.nii.gz'))
    nib.save(lab, os.path.join(path_label, f'Patient{index}.nii.gz'))

    print(f'patient_generated_{index} is saved', end='\r')   


output_path = 'finalResults'

name_folder = 'paperimages90'
os.mkdir(os.path.join(output_path, name_folder))
output = os.path.join(output_path, name_folder)
check_ds = Dataset(data=train_files, transform=generat_transforms2)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
for index, patient in enumerate(check_loader):
    index =index+1
    save_nifti(patient['image'], patient['label'], output, index)
print(f'step {i} done')
