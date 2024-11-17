# Load and prepare the data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Simple UNet-like model for denoising
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

# Forward diffusion function to add noise
def add_noise(x, noise_level):
    noise = noise_level * torch.randn_like(x)
    return x + noise, noise

# Reverse process (denoising)
def reverse_process(model, noisy_image, num_steps, noise_level):
    for step in range(num_steps):
        pred_noise = model(noisy_image)
        noisy_image = noisy_image - noise_level * pred_noise
    return noisy_image

def load_and_prepare_data(file_path):
    mat_data = sio.loadmat(file_path)
    kspace_data = mat_data['kData']

    # Convert to PyTorch tensor and complex format
    kspace_tensor = torch.from_numpy(kspace_data).float()
    if kspace_tensor.shape[-1] == 2:
        kspace_tensor = torch.view_as_complex(kspace_tensor)
    
    # Apply inverse FFT to get the image
    image_space = torch.fft.ifftn(kspace_tensor, dim=(-2, -1), norm="ortho")
    image_magnitude = torch.abs(image_space)

    # Select a specific slice (e.g., first phase and first coil)
    image_magnitude = image_magnitude[:, :, 0, 0]  # Adjust indices as needed

    # Add channel and batch dimensions
    image_magnitude = image_magnitude.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, height, width]

    return image_magnitude

# Modify the training function to work with 4D input
def train_diffusion_model(train_data, num_epochs=100, lr=1e-4, noise_level=0.1, num_steps=5):
    model = SimpleUNet(in_channels=1, out_channels=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass: add noise
        noisy_image, noise = add_noise(train_data, noise_level)
        
        # Predict noise and calculate loss
        pred_noise = model(noisy_image)
        loss = criterion(pred_noise, noise)
        
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    
    return model

# Example usage
file_path = 'C:\\Users\\carol\\Desktop\\UGthesis_cMRIrecon\\unsupervised_MRIrecon\\matlab_scripts\\new_fs_ocmr\\fs_0001_1_5T.mat'
train_data = load_and_prepare_data(file_path)

# Train the model
trained_model = train_diffusion_model(train_data, num_epochs=50)
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


