import os
import numpy as np
import nibabel as nib

arr = np.random.rand(128, 128, 64).astype(np.float32)

affine = np.eye(4)
img = nib.Nifti1Image(arr, affine)

save_path = "../data/nifti_dummy/dummy.nii.gz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
nib.save(img, save_path)
