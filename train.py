import torch
import torch.nn as nn
import torch.nn.functional as F


def train_ddim(model, train_data, num_epochs=100, lr=1e-4, batch_size=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        for batch in train_data:
            batch = batch.to(model.device)
            t = torch.randint(0, model.timesteps, (batch.size(0),), device=model.device)
            
            x_t, noise = model.forward_diffusion(batch, t)
            pred_noise = model.unet(x_t, t)
            
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")