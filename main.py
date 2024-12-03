import os
from matplotlib import pyplot as plt
from custom_dataset import CustomDataset
from GRDN import GRDN
from SelfEnsembleDenoisingAutoEncoder import SelfEnsembleDenoisingModel

dataset = CustomDataset('dataset.csv')
train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=4)

# Hyperparameters
in_channels = 3
out_channels = 64
growth_rate = 32
num_layers = 4
num_rdb_blocks = 3
num_grdb_blocks = 5
num_epochs = 15
learning_rate = 0.001


def train():
    model = GRDN(in_channels, out_channels, growth_rate, num_layers, num_rdb_blocks, num_grdb_blocks)
    model.train_model(train_loader, val_loader, num_epochs, learning_rate)
    model.save_model('grdn_model.pth')


def test():
    model = GRDN.load_model('grdn_model.pth')
    model.test_model(test_loader, save_path='./outputs/test_results.png')
    # ensemble_model = SelfEnsembleDenoisingModel(model)
    # print('Ensemble Model test')
    # ensemble_model.test_model(test_loader, save_path='./outputs/ensemble_test_results.png')


def predict(dir_name:str|int = '0001_001_S6_00100_00060_3200_L'):
    if isinstance(dir_name, str) and not os.path.exists(f'SIDD_Small_sRGB_Only/Data/{dir_name}'):
        raise ValueError('Invalid directory name')
    if isinstance(dir_name, int):
        dir_name_start = str(dir_name).zfill(4)
        for d in os.listdir('SIDD_Small_sRGB_Only/Data'):
            if d.startswith(dir_name_start):
                dir_name = d
                break
        if isinstance(dir_name, int):
            raise ValueError('Invalid directory name')
    def plot_predict(noisy_image, gt_image, denoised_image):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        noisy_image = plt.imread(noisy_image)
        gt_image = plt.imread(gt_image)
        denoised_image = plt.imread(denoised_image)
        axes[0].imshow(noisy_image)
        axes[0].set_title('Noisy Image')
        axes[1].imshow(gt_image)
        axes[1].set_title('Clean Image')
        axes[2].imshow(denoised_image)
        axes[2].set_title('Denoised Image')
        for ax in axes:
            ax.axis('off')
        plt.show()
    base_model = GRDN.load_model('grdn_model.pth')
    ensemble_model = SelfEnsembleDenoisingModel(base_model)

    noisy_image = f'SIDD_Small_sRGB_Only/Data/{dir_name}/NOISY_SRGB_010.PNG'
    gt_image = f'SIDD_Small_sRGB_Only/Data/{dir_name}/GT_SRGB_010.PNG'
    denoised_image = './outputs/denoised.PNG'
    base_model.denoise_image(noisy_image, save_path=denoised_image)
    print('Denoised image saved at:', denoised_image)

    plot_predict(noisy_image, gt_image, denoised_image)


if __name__ == '__main__':
    train()
