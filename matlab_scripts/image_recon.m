k_space = load('cine_lax_ks.mat');
centered_kspace = load('cine_lax_calib.mat');
k_space = k_space.Recon_ks;
centered = centered_kspace.Calib; 
%%

sz = size(k_space);
kx_ky = k_space(:,:,1, 1, 1); % kx and ky and slice numbers 
imagesc(abs(kx_ky))

 %%
% Apply inverse FFT along spatial dimensions (kx, ky)
image_space = ifftshift(ifft2(ifftshift(kx_ky)));

% Combine coil images using sum-of-squares
final_image = sqrt(sum(abs(image_space).^2, 3));  % Combine along coil dimension

% Visualize reconstructed images for each time frame
for t = 1:size(final_image, 3)
    figure;
    imshow(abs(final_image(:,:,t)), []);  % Display the t-th time frame
    title(['Reconstructed Image at Time Frame: ', num2str(t)]);
end
