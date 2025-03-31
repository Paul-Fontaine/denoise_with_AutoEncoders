import torch
import torch.nn as nn
import torch.optim as optim
from numpy import ndarray
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os
from custom_dataset import val_transform, inverse_transform


class AutoEncoderBaseClass(nn.Module):
    def __init__(self):
        super(AutoEncoderBaseClass, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()

    def train_model(self, train_loader, val_loader, num_epochs=15, learning_rate=0.0001, save_path='./outputs/training_metrics_evolution.png'):
        # print device
        print(f"Training on {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        # print number of parameters and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        # print with thousands separator
        print(f"Total Parameters: {total_params:,}")
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_trainable_params:,}")

        self.to(self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            for noisy_imgs, gt_imgs_transformed, gt_imgs in train_bar:
                noisy_imgs, gt_imgs = noisy_imgs.to(self.device), gt_imgs.to(self.device)
                optimizer.zero_grad()
                outputs = self(noisy_imgs)
                loss = self.criterion(outputs, gt_imgs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            self.eval()
            val_loss = 0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            with torch.no_grad():
                for noisy_imgs, gt_imgs in val_bar:
                    noisy_imgs, gt_imgs = noisy_imgs.to(self.device), gt_imgs.to(self.device)
                    outputs = self(noisy_imgs)
                    loss = self.criterion(outputs, gt_imgs)
                    val_loss += loss.item()
                    val_bar.set_postfix(loss=loss.item())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        # Plot training and validation loss evolution
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.xticks(range(1, num_epochs+1))
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Evolution")
        plt.savefig(save_path)
        plt.show()

    def test_model(self, test_loader, save_path='./outputs/test_results.png'):
        self.to(self.device)
        self.eval()
        psnr_scores = []
        ssim_scores = []
        results_dir = os.path.dirname(save_path)
        os.makedirs(results_dir, exist_ok=True)
        figure, axes = plt.subplots(3, 5, figsize=(20, 8))
        for ax in axes.flatten():
            ax.axis("off")

        with torch.no_grad():
            for idx, (noisy_imgs, gt_imgs) in enumerate(tqdm(test_loader, desc="Testing")):
                noisy_imgs, gt_imgs = noisy_imgs.to(self.device), gt_imgs.to(self.device)
                outputs = self(noisy_imgs)

                # Save sample results
                if idx < 5:  # Save first 5 examples
                    axes[0, idx].imshow(inverse_transform(noisy_imgs[0]))
                    axes[0, idx].set_title("Noisy Image")
                    axes[1, idx].imshow(inverse_transform(outputs[0]))
                    axes[1, idx].set_title("Denoised Image")
                    axes[2, idx].imshow(inverse_transform(gt_imgs[0]))
                    axes[2, idx].set_title("Ground Truth")

                # Compute PSNR
                mse = torch.mean((outputs - gt_imgs) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(1 / mse)
                psnr_scores.extend(psnr.tolist())

                # Compute SSIM
                for output, gt in zip(outputs, gt_imgs):
                    output_img = inverse_transform(output)
                    gt_img = inverse_transform(gt)

                    # Compute SSIM
                    ssim_value = ssim(gt_img, output_img, data_range=1, channel_axis=2, multichannel=True)
                    ssim_scores.append(ssim_value)

        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")

        figure.suptitle(f"Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def denoise_image(self, image, save_path=None):
        self.to(self.device)
        self.eval()

        # Load and preprocess image
        if isinstance(image, str):  # If path is provided
            image = Image.open(image).convert("RGB")

        input_image = val_transform(image).unsqueeze(0).to(self.device)

        # Denoise
        with torch.no_grad():
            output_image = self(input_image)

        # Postprocess
        output_image = output_image.squeeze().cpu()
        output_image = inverse_transform(output_image)
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)

        if save_path:
            output_image.save(save_path)
        return output_image

    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load_model(path):
        # disable future warnings
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        return torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=False)
