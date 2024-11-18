
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os 

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def add_noise(x, noise_level):
    """
    Add Gaussian noise to both real and imaginary parts of the k-space data.
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, 2, kx, ky], 
                          where the second dimension represents real and imaginary parts.
        noise_level (float): Standard deviation of the Gaussian noise to add.
    Returns:
        torch.Tensor: Noisy data of the same shape as the input.
        torch.Tensor: Noise that was added to the input.
    """
    noise = noise_level * torch.randn_like(x)  # Generate noise with the same shape as x
    noisy_x = x + noise  # Add noise to the input
    return noisy_x, noise

# Reverse process (denoising) 
def reverse_process(model, noisy_image, num_steps, noise_level):
    for step in range(num_steps):
        pred_noise = model(noisy_image)
        noisy_image = noisy_image - noise_level * pred_noise
    return noisy_image

def train_diffusion_model(train_data, num_epochs=100, lr=1e-4, noise_level=0.1, num_steps=5):
    """
    Train the diffusion model on k-space data with noise added to both real and imaginary parts.
    Args:
        train_data (torch.Tensor): Input tensor of shape [batch_size, 2, kx, ky].
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        noise_level (float): Standard deviation of the Gaussian noise.
        num_steps (int): Number of denoising steps (for reverse process).
    Returns:
        SimpleUNet: Trained model.
    """
    model = SimpleUNet(in_channels=2, out_channels=2)  # Model for 2-channel input/output
    criterion = nn.MSELoss()  # Loss function to compare predicted and actual noise
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass: add noise to the k-space data
        noisy_kspace, noise = add_noise(train_data, noise_level)

        # Predict noise using the model
        pred_noise = model(noisy_kspace)

        # Calculate the loss between predicted noise and the actual noise
        loss = criterion(pred_noise, noise)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model

def load_and_prepare_data(folder_path):
    """
    Load and prepare data from all .mat files in a folder containing 5D k-space data.
    Args:
        folder_path (str): Path to the folder containing .mat files.
        slice_idx (int): Index of the slice to extract.
        coil_idx (int): Index of the coil to extract.
        phase_idx (int): Index of the phase to extract.
    Returns:
        torch.Tensor: A tensor containing all the prepared image data with shape [num_samples, 1, height, width].
    """
    data_list = []
    
    # Iterate through all .mat files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            full_path = os.path.join(folder_path, file_name)
            mat_data = sio.loadmat(full_path)
            
            # Extract the 5D k-space data
            kspace_data = mat_data['kData']  # Shape: [kx, ky, slice, coil, phase]

            real_part = torch.from_numpy(kspace_data.real).float()
            imag_part = torch.from_numpy(kspace_data.imag).float()
            kspace_tensor = torch.complex(real_part, imag_part) 

            # Add batch dimension
            kspace_tensor = kspace_tensor.unsqueeze(0)  # Shape: [1, 2, kx, ky]

            # Append to the list
            data_list.append(kspace_tensor)
    # Stack all loaded data into a single tensor
    return torch.cat(data_list, dim=0)  # Shadata_list, dim=0)  # Shape: [num_samples, 1, height, width]

# Example usage
file_path = 'C:\\Users\\carol\\Desktop\\UGthesis_cMRIrecon\\unsupervised_MRIrecon\\matlab_scripts\\new_us_ocmr'
train_data = load_and_prepare_data(file_path)

# Train the model
trained_model = train_diffusion_model(train_data, num_epochs=1)
# Test the model with a noisy image and visualize the output
noisy_image, _ = add_noise(train_data, noise_level=0.1)
reconstructed_image = reverse_process(trained_model, noisy_image, num_steps=5, noise_level=0.1)

# Visualize the input, noisy, and reconstructed images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(train_data[0, 0].numpy(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image[0, 0].detach().numpy(), cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image[0, 0].detach().numpy(), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()


