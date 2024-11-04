import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from data_augment import AugmentationPipeline  # Ensure the import matches your directory structure

# Define a simple hparams class with necessary parameters
class HParams:
    aug_weight_translation = 1.0
    aug_weight_rotation = 1.0
    aug_weight_scaling = 1.0
    aug_weight_shearing = 1.0
    aug_weight_rot90 = 0.5
    aug_weight_fliph = 0.5
    aug_weight_flipv = 0.5
    aug_upsample = False
    aug_upsample_factor = 2
    aug_upsample_order = 1
    aug_interpolation_order = 1
    aug_max_translation_x = 0.1
    aug_max_translation_y = 0.1
    aug_max_rotation = 15.0
    aug_max_scaling = 0.2
    aug_max_shearing_x = 10.0
    aug_max_shearing_y = 10.0
    max_train_resolution = None  # Adjust if needed

# Initialize the AugmentationPipeline
hparams = HParams()
augmentation_pipeline = AugmentationPipeline(hparams)

# Function to apply the inverse FFT and visualize the augmented image
def visualize_augmentation_pipeline(augmentation_pipeline, kspace, target_size):
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
    # Convert to NumPy for FFT processing
    kspace_tensor = kspace_tensor.numpy()

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

# Load .mat file
mat_data = sio.loadmat('C:\\Users\\carol\\Desktop\\UGthesis_cMRIrecon\\unsupervised_MRIrecon\\matlab_scripts\\new_fs_ocmr\\fs_0001_1_5T.mat')
visualize_augmentation_pipeline(augmentation_pipeline, mat_data, mat_data)


