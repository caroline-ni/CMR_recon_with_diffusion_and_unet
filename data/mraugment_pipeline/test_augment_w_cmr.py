import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.fft import fftshift, ifftshift, ifft2
from data_augment import AugmentationPipeline  # Ensure the import matches your directory structure

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

# Function to apply the inverse FFT and visualize the augmented image
def visualize_augmentation_pipeline(augmentation_pipeline, kspace):
    """
    Visualizes augmented MRI images using the AugmentationPipeline.
    """
    # Load k-space data from the dictionary
    kspace_data = kspace['kData']  # Replace with actual key if different

    # Define dimensions to transform (kx and ky) and desired output shape
    dim = (0, 1)  # Transform along kx and ky dimensions
    img_shape = kspace_data.shape[:2]  # Use k-space dimensions for image shape
    image_space_data = transform_kspace_to_image(kspace_data, dim=dim, img_shape=img_shape)

    augmented_image = augmentation_pipeline.augment_image(image_space_data)

    if isinstance(augmented_image, torch.Tensor):
        augmented_image = augmented_image.detach().cpu().numpy()

    # Coil combination using root-sum-of-squares (RSS)
    final_image = np.sqrt(np.sum(np.abs(augmented_image) ** 2, axis=3))  # Sum over coil dimension
    
    # Visualize the reconstructed image at the first slice and phase (adjust indices as needed)
    plt.imshow(np.abs(final_image[:, :, 0, 0]), cmap='gray')
    plt.title(f'Slice 0, Phase 0')
    plt.axis('off')
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

    hparams = HParams(
        aug_weight_translation=0,  # No translation
        aug_weight_rotation=1,     # Enable arbitrary rotations (e.g., 45°)
        aug_weight_scaling=0,      # No scaling
        aug_weight_shearing=0,     # No shearing
        aug_weight_rot90=1,        # Enable 90-degree rotations
        aug_weight_fliph=0,        # No horizontal flip
        aug_weight_flipv=0,        # No vertical flip
        aug_max_translation_x=0,   # Translation disabled
        aug_max_translation_y=0,   # Translation disabled
        aug_max_rotation=45,       # Maximum arbitrary rotation angle is ±45°
        aug_max_scaling=0,         # Scaling disabled
        aug_max_shearing_x=0,      # Shearing disabled
        aug_max_shearing_y=0,      # Shearing disabled
        aug_upsample=False         # No upsampling needed
    )
    # Initialize the pipeline
    augmentation_pipeline = AugmentationPipeline(hparams)

    # Set augmentation strength to 1 (ensures all enabled augmentations are applied)
    augmentation_pipeline.set_augmentation_strength(1)

    visualize_augmentation_pipeline(augmentation_pipeline, mat_data)

    # visualize_pipeline(mat_data)
    # sigpy_reconstruct(mat_data)
    # kspace_data = mat_data['kData']
    # print(kspace_data.dtype)

