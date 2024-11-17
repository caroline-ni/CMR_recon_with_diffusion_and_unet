import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.fft import fftshift, ifftshift, ifft2
from data_augment import AugmentationPipeline  # Ensure the import matches your directory structure
import math
import sigpy as sp

class HParams:
    def __init__(self,
                 aug_weight_translation=0,
                 aug_weight_rotation=0,
                 aug_weight_scaling=0,
                 aug_weight_shearing=0,
                 aug_weight_rot90=0,
                 aug_weight_fliph=0,
                 aug_weight_flipv=0,
                 aug_upsample=False,
                 aug_upsample_factor=2,
                 aug_upsample_order=1,
                 aug_interpolation_order=1,
                 aug_max_translation_x=0,
                 aug_max_translation_y=0,
                 aug_max_rotation=0,
                 aug_max_scaling=0,
                 aug_max_shearing_x=0,
                 aug_max_shearing_y=0,
                 max_train_resolution=None):
        self.aug_weight_translation = aug_weight_translation
        self.aug_weight_rotation = aug_weight_rotation
        self.aug_weight_scaling = aug_weight_scaling
        self.aug_weight_shearing = aug_weight_shearing
        self.aug_weight_rot90 = aug_weight_rot90
        self.aug_weight_fliph = aug_weight_fliph
        self.aug_weight_flipv = aug_weight_flipv
        self.aug_upsample = aug_upsample
        self.aug_upsample_factor = aug_upsample_factor
        self.aug_upsample_order = aug_upsample_order
        self.aug_interpolation_order = aug_interpolation_order
        self.aug_max_translation_x = aug_max_translation_x
        self.aug_max_translation_y = aug_max_translation_y
        self.aug_max_rotation = aug_max_rotation
        self.aug_max_scaling = aug_max_scaling
        self.aug_max_shearing_x = aug_max_shearing_x
        self.aug_max_shearing_y = aug_max_shearing_y
        self.max_train_resolution = max_train_resolution

# Create an instance with custom hyperparameters
hparams = HParams(aug_weight_translation=0, aug_max_rotation=0, aug_weight_fliph=0)

augmentation_pipeline = AugmentationPipeline(hparams)

# Function to apply the inverse FFT and visualize the augmented image
def visualize_augmentation_pipeline(augmentation_pipeline, kspace):
    """
    Visualizes augmented MRI images using the AugmentationPipeline.
    """
    # Assuming the k-space data is stored under the key 'kData'
    kspace_data = kspace['kData']  # Replace with the actual key if different

    # Convert k-space data to PyTorch tensor
    kspace_tensor = torch.from_numpy(kspace_data).float()

    # Check if the k-space tensor needs conversion to complex form
    if kspace_tensor.shape[-1] == 2:  # Last dimension is real + imaginary
        kspace_tensor = torch.view_as_complex(kspace_tensor)  # Convert to complex-valued tensor

    # Initialize an empty array for the reconstructed images
    reconstructed_images = np.zeros(kspace_tensor.shape, dtype=np.complex64)

    # Apply augmentations
    augmented_image = augmentation_pipeline.augment_image(kspace_tensor)

    # Perform IFFT across kx and ky for a specific coil, time, and slice
    image_space = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(augmented_image[:, :, :, 1, 1], axes=(0, 1)),
            axes=(0, 1)
        ),
        axes=(0, 1)
    )

    # Combine coils using Sum of Squares (SoS)
    combined_img = np.sqrt(np.sum(np.abs(image_space) ** 2, axis=2))

    # Debug: Print shape after augmentation and before conversion to magnitude
    print(f"Shape after coil combination: {combined_img.shape}")

    # Store in the reconstructed images array (example: time frame 1, slice 1)
    reconstructed_images[:, :, 0, 1, 1] = combined_img

    # Convert to magnitude
    reconstructed_images = np.abs(reconstructed_images)

    # Visualize the selected image
    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_images[:, :, 0, 1, 1], cmap='gray')  # Display 2D slice
    plt.axis('off')
    plt.title('Reconstructed Image')
    plt.show()


def visualize_pipeline(kspace):
    """
    Visualizes augmented MRI images from k-space data.
    
    Parameters:
    - kspace: Dictionary containing k-space data under the key 'kData'.
    - coil_number: Coil index to use if inspecting individual coils.
    """
    # Load k-space data from the dictionary
    kspace_data = kspace['kData']  # Replace with actual key if different
    print(f"K-space data shape: {kspace_data.shape}")

    # Define dimensions to transform (kx and ky) and desired output shape
    dim = (0, 1)  # Transform along kx and ky dimensions
    img_shape = kspace_data.shape[:2]  # Use k-space dimensions for image shape

    # Transform k-space to image space
    image_space_data = transform_kspace_to_image(kspace_data, dim=dim, img_shape=img_shape)

    # Coil combination using root-sum-of-squares (RSS)
    final_image = np.sqrt(np.sum(np.abs(image_space_data) ** 2, axis=3))  # Sum over coil dimension

    # Visualize the reconstructed image at the first slice and phase (adjust indices as needed)
    plt.imshow(np.abs(final_image[:, :, 0, 0]), cmap='gray')
    plt.title(f'Slice 0, Phase 0')
    plt.axis('off')
    plt.show()

# helper function 
def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    # Set default dimensions to transform (all dimensions by default)
    if dim is None:
        dim = range(k.ndim)

    # Apply fftshift, inverse FFT, and ifftshift
    img = np.fft.fftshift(k, axes=dim)
    img = np.fft.ifftn(img, s=img_shape, axes=dim)
    img = np.fft.ifftshift(img, axes=dim)

    # Scale the image as in MATLAB's ifft2
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


if __name__ == "__main__":
    # Load .mat file
    mat_data = sio.loadmat('C:\\Users\\carol\\Desktop\\UGthesis_cMRIrecon\\unsupervised_MRIrecon\\matlab_scripts\\new_fs_ocmr\\fs_0001_1_5T.mat')
    file = "C:\\Users\\carol\\Desktop\\UGthesis_cMRIrecon\\unsupervised_MRIrecon\\matlab_scripts\\fs\\fs_0001_1_5T.h5"
    # visualize_augmentation_pipeline(augmentation_pipeline, mat_data)

    visualize_pipeline(mat_data)
    # sigpy_reconstruct(mat_data)
    # kspace_data = mat_data['kData']
    # print(kspace_data.dtype)

