import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from scipy.ndimage import maximum_filter

class DATASET(Dataset):
    def __init__(self, label_list, dataset_list):
        self.label_list = label_list
        self.dataset_list = dataset_list
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        img_dir = self.dataset_list[idx]
        img = pilimg.open(img_dir)
        img = T.ToTensor()(img).float()
        img = (img - img.mean()) / img.std()
        label = img_dir.split("/")[-2]
        label = self.label_list.index(label)
        return img, label

def show_dataset(dataset_dir, label_list, show_img_num):

    fig, ax = plt.subplots(len(label_list), show_img_num)
    for label_name in label_list:
        data_list = os.listdir(f"{dataset_dir}/{label_name}")
        for i in range(show_img_num):
            example_data_dir = f"{dataset_dir}/{label_name}/{data_list[i]}"
            example_img = pilimg.open(example_data_dir)
            example_img = np.array(example_img)
            ax[label_list.index(label_name), i].set_title(label_name)
            ax[label_list.index(label_name), i].imshow(example_img, cmap = "gray")
            ax[label_list.index(label_name), i].axis('off')

    plt.show()


def get_dataloader(dataset_dir, label_list, train_valid_ratio, batch_size):

    dataset_list = []
    for label in label_list:
        data_list = os.listdir(f"{dataset_dir}/{label}")
        for data in data_list:
            dataset_list.append(f"{dataset_dir}/{label}/{data}")

    np.random.shuffle(dataset_list)
    train_dataset_list = dataset_list[:int(train_valid_ratio[0]*len(dataset_list)/(train_valid_ratio[0]+train_valid_ratio[1]))]
    valid_dataset_list = dataset_list[int(train_valid_ratio[0]*len(dataset_list)/(train_valid_ratio[0]+train_valid_ratio[1])):]


    train_dataset = DATASET(label_list, train_dataset_list)
    valid_dataset = DATASET(label_list, valid_dataset_list)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle = False)
    
    return train_dataloader, valid_dataloader

def train(model, train_dataloader, valid_dataloader, criterion, optimizer, device):
    model.train()
    model.to(device)
    train_loss = 0
    train_acc = 0
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        pred = torch.argmax(pred, dim = 1)
        train_loss += loss.item()
        train_acc += (pred == y).sum().item()

    train_loss = train_loss / (i + 1)
    train_acc = train_acc * 100 / len(train_dataloader.dataset)

    with torch.no_grad():
        valid_loss = 0
        valid_acc = 0
        model.eval()
        for i, (x, y) in enumerate(valid_dataloader):
            x = x.to(device).float()
            y = y.to(device).long()
            pred = model(x)
            loss = criterion(pred, y)
            pred = torch.argmax(pred, dim = 1)
            valid_loss += loss.item()
            valid_acc += (pred == y).sum().item()

        valid_loss = valid_loss / (i + 1)
        valid_acc = valid_acc * 100 / len(valid_dataloader.dataset)

    print(f"| Training Loss : {train_loss:.3f} | Training Accuracy : {train_acc:.3f} % |")
    print(f"| Validation Loss : {valid_loss:.3f} | Validation Accuracy : {valid_acc:.3f} % |")

    return train_loss, train_acc, valid_loss, valid_acc

def get_atom_pos(data_sig):

    pca = PCA(n_components = 5)
    W = pca.fit_transform(data_sig)
    H = pca.components_
    data_sig = np.matmul(W, H) + data_sig.mean(axis = 0)

    atom_position = np.array(np.where(maximum_filter(data_sig, size = 17) == data_sig)).T
    atom_position = atom_position[(atom_position[:, 0] - 24 > 0)&(atom_position[:, 0] + 24 < 1024)&(atom_position[:, 1] - 24 > 0)&(atom_position[:, 1] + 24 < 1024)]

    return atom_position

def get_test_img(test_data_dir):
    data_sig = pilimg.open(test_data_dir)
    data_sig = np.array(data_sig)
    atom_position = get_atom_pos(data_sig)
    atom_img_list = []

    for i in range(len(atom_position)):
        atom_img = data_sig[atom_position[i, 0] - 24 : atom_position[i, 0] + 24,
                            atom_position[i, 1] - 24 : atom_position[i, 1] + 24
                            ]
        atom_img = (atom_img - atom_img.mean()) / atom_img.std()
        atom_img_list.append(atom_img)

    atom_img_list = np.array(atom_img_list)
    atom_img_list = torch.from_numpy(atom_img_list)
    atom_img_list = atom_img_list.view(-1, 1, 48, 48)

    data_sig = data_sig.astype(np.uint8)

    return data_sig, atom_position, atom_img_list

def map_to_rgb8(mapping_result):
    h, w = mapping_result.shape
    color_mapping = np.zeros((h, w, 3))

    color_mapping[np.where(mapping_result == 1)[0], np.where(mapping_result == 1)[1], 0] = 255

    color_mapping[np.where(mapping_result == 2)[0], np.where(mapping_result == 2)[1], 0] = 255
    color_mapping[np.where(mapping_result == 2)[0], np.where(mapping_result == 2)[1], 1] = 255

    color_mapping[np.where(mapping_result == 3)[0], np.where(mapping_result == 3)[1], 1] = 255

    color_mapping[np.where(mapping_result == 4)[0], np.where(mapping_result == 4)[1], 1] = 255
    color_mapping[np.where(mapping_result == 4)[0], np.where(mapping_result == 4)[1], 2] = 255

    color_mapping[np.where(mapping_result == 5)[0], np.where(mapping_result == 5)[1], 2] = 255

    color_mapping = color_mapping.astype(np.uint8)

    return color_mapping

def test(test_data_dir, model, device, show_result = False):
    data_sig, atom_position, atom_img_list = get_test_img(test_data_dir)
    h, w = data_sig.shape
    with torch.no_grad():
        model.to(device)
        model.eval()
        atom_img_list = atom_img_list.to(device).float()
        pred = model(atom_img_list)
        pred = torch.argmax(pred, dim = 1)
        pred = pred.cpu().detach().numpy()
        mapping_result = np.zeros((h, w))
        r = 6
        for i in range(-r, r+1):
            for j in range(-r , r+1):
                if i**2 + j**2 <= r**2:
                    mapping_result[atom_position[:, 0] + i, atom_position[:, 1] + j] = pred + 1
        
        color_map = map_to_rgb8(mapping_result)

        if show_result:
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title("STEM HAADF Image")
            ax[0].imshow(data_sig, cmap = 'gray')
            ax[0].axis('off')
            ax[1].set_title("Mapping Image")
            ax[1].imshow(color_map)
            ax[1].axis('off')
            plt.show()
        
        W_count = (pred == 0).sum().item()
        VW_count = (pred == 1).sum().item()
        Se2_count = (pred == 2).sum().item()
        VSe_count = (pred == 3).sum().item()
        VSe2_count = (pred == 4).sum().item()
    
    print(f"DATA : {test_data_dir}")
    print(f"| W : {W_count} | VW : {VW_count} | Se2 : {Se2_count} | VSe : {VSe_count} | VSe2 : {VSe2_count} |")
    print(f"| V dopants : {VW_count * 100 / (W_count + VW_count + 2 * (Se2_count + VSe_count + VSe2_count)):.3f} % |")
    print(f"| Se vacancies : {(VSe_count + VSe2_count * 2) * 100 / (W_count + VW_count + 2 * (Se2_count + VSe_count + VSe2_count)):.3f} % |")


