import os
import pandas as pd

data_root = './SIDD_Small_sRGB_Only/Data'

df = pd.DataFrame(columns=['noisy', 'gt'])
noisy_paths = []
gt_paths = []

for dir in os.listdir(data_root):
    for file in os.listdir(os.path.join(data_root, dir)):
        if file.startswith('NOISY'):
            noisy_path = os.path.join(data_root, dir, file)
            gt_path = noisy_path.replace('NOISY', 'GT')
        elif file.startswith('GT'):
            gt_path = os.path.join(data_root, dir, file)
            noisy_path = gt_path.replace('GT', 'NOISY')
        else:
            print('Invalid file name: ', file)
            continue
        noisy_paths.append(noisy_path)
        gt_paths.append(gt_path)

df['noisy'] = noisy_paths
df['gt'] = gt_paths
df.to_csv('dataset.csv', index=False)
