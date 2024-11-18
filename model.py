import torch
import torch.nn as nn
import torch.nn.functional as F

class DDIM(nn.Module):
    def __init__(self, unet, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        """
        Initialize DDIM with a U-Net model and diffusion schedule.
        Args:
            unet (nn.Module): U-Net model for noise prediction.
            timesteps (int): Number of diffusion steps.
            beta_start (float): Starting value of beta (noise schedule).
            beta_end (float): Ending value of beta (noise schedule).
            device (str): 'cpu' or 'cuda' for computations.
        """
        super(DDIM, self).__init__()
        self.unet = unet.to(device)
        self.timesteps = timesteps
        self.device = device

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x_start, t):
        """
        Apply forward diffusion to add noise to the data.
        Args:
            x_start (torch.Tensor): Clean k-space data.
            t (torch.Tensor): Timestep indices.
        Returns:
            x_t (torch.Tensor): Noisy k-space data at time t.
            noise (torch.Tensor): Added noise.
        """
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None, None]
        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_t, noise

    def reverse_diffusion(self, x_t, t, eta=0):
        """
        Perform reverse diffusion to denoise the k-space data.
        Args:
            x_t (torch.Tensor): Noisy k-space data at time t.
            t (torch.Tensor): Timestep indices.
            eta (float): Noise scaling factor (0 for deterministic DDIM).
        Returns:
            x_pred (torch.Tensor): Predicted k-space data at time t-1.
        """
        pred_noise = self.unet(x_t, t)
        alpha = self.alphas[t][:, None, None, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None, None, None]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t][:, None, None, None, None, None]

        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip predictions

        pred_mean = torch.sqrt(alpha_cumprod_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
        if eta > 0:
            noise = torch.randn_like(x_t)
            pred_x0 = pred_mean + eta * torch.sqrt(self.posterior_variance[t])[:, None, None, None, None, None] * noise
        
        return pred_mean

    def sample(self, shape, eta=0):
        """
        Generate samples using DDIM sampling.
        Args:
            shape (tuple): Shape of the output k-space data.
            eta (float): Noise scaling factor (0 for deterministic DDIM).
        Returns:
            x_0 (torch.Tensor): Generated k-space data.
        """
        x_t = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x_t = self.reverse_diffusion(x_t, t_tensor, eta=eta)
        return x_t