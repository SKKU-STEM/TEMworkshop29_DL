import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import torch
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from scipy.ndimage import maximum_filter
from sklearn.metrics import confusion_matrix


class MODEL(nn.Module):
    def __init__(self, img_channels, conv_num_features1, conv_num_features2, conv_num_features3,
                 fc_num_features, label_list):
        super(MODEL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = img_channels,
                               out_channels = conv_num_features1,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True)
        self.conv2 = nn.Conv2d(in_channels = conv_num_features1,
                               out_channels = conv_num_features2,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True)
        self.conv3 = nn.Conv2d(in_channels = conv_num_features2,
                               out_channels = conv_num_features3,
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               bias = True)

        self.fc1 = nn.Linear(conv_num_features3 * 6 * 6, fc_num_features)
        self.fc2 = nn.Linear(fc_num_features, len(label_list))

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(len(x), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x



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

    fig, ax = plt.subplots(len(label_list), show_img_num, figsize = (15, 9))
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

def train(model, train_dataloader, valid_dataloader, loss_function, optimizer, device):
    model.train()
    model.to(device)
    train_loss = 0
    train_acc = 0
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_function(pred, y)
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
            loss = loss_function(pred, y)
            pred = torch.argmax(pred, dim = 1)
            valid_loss += loss.item()
            valid_acc += (pred == y).sum().item()

        valid_loss = valid_loss / (i + 1)
        valid_acc = valid_acc * 100 / len(valid_dataloader.dataset)

    print(f"| Training Loss : {train_loss:.3f} | Training Accuracy : {train_acc:.3f} % |")
    print(f"| Validation Loss : {valid_loss:.3f} | Validation Accuracy : {valid_acc:.3f} % |")

    return train_loss, train_acc, valid_loss, valid_acc


def show_train_graph(training_log):

    training_log = np.array(training_log)
    
    plt.plot(training_log[:, 0], training_log[:, 1], label = "Training")
    plt.plot(training_log[:, 0], training_log[:, 3], label = "Validation")
    plt.legend()
    plt.title("Loss graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    
    plt.plot(training_log[:, 0], training_log[:, 2], label = "Training")
    plt.plot(training_log[:, 0], training_log[:, 4], label = "Validation")
    plt.legend()
    plt.title("Accuracy graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()



def get_atom_pos(data_sig):

    pca = PCA(n_components = 2)
    W = pca.fit_transform(data_sig)
    H = pca.components_
    data_sig = np.matmul(W, H) + data_sig.mean(axis = 0)

    atom_position = np.array(np.where(maximum_filter(data_sig, size = 19) == data_sig)).T
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

    color_mapping[np.where(mapping_result == 3)[0], np.where(mapping_result == 3)[1], 1] = 128

    color_mapping[np.where(mapping_result == 4)[0], np.where(mapping_result == 4)[1], 1] = 255
    color_mapping[np.where(mapping_result == 4)[0], np.where(mapping_result == 4)[1], 2] = 255

    color_mapping[np.where(mapping_result == 5)[0], np.where(mapping_result == 5)[1], 2] = 255

    color_mapping = color_mapping.astype(np.uint8)

    return color_mapping

def plot_result(data_sig, mapping_result, pred, test_data_dir):
    fig, ax = plt.subplots(1, 2, figsize = (15, 9))
    ax[0].set_title("STEM HAADF Image", fontsize = 24)
    ax[0].imshow(data_sig, cmap = 'gray')
    ax[0].axis('off')
    ax[1].set_title("Mapping Image", fontsize = 24)
    ax[1].imshow(mapping_result)
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

def save_map(test_data_dir, mapping_result):
    dir_list = test_data_dir.split("/")
    mapping_dir = ""
    for i in range(len(dir_list) - 1):
        mapping_dir += f"{dir_list[i]}/"
    mapping_dir = f"{mapping_dir[:-1]}_result"
    try:
        os.mkdir(mapping_dir)
    except:
        pass
    mapping_dir = f"{mapping_dir}/{dir_list[-1]}"

    color_map = pilimg.fromarray(mapping_result)
    color_map.save(mapping_dir)


def test(test_data_dir, model, device, show_result = False, save_mapping_result = False):
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
        
        mapping_result = map_to_rgb8(mapping_result)

        if show_result:
            plot_result(data_sig, mapping_result, pred, test_data_dir)
        if save_mapping_result:
            save_map(test_data_dir, mapping_result)

    test_result = np.concatenate((atom_position, pred.reshape(-1, 1)), axis = 1)
    return test_result, mapping_result

def evaluation(test_data_dir, test_result):
    dir_list = test_data_dir.split("/")
    src_folder = ""
    for i in range(len(dir_list) - 2):
        src_folder += f"{dir_list[i]}/"
    src_folder = src_folder[:-1]
    img_name = dir_list[-1]
    gt_data_dir = f"{src_folder}/ground_truth/{img_name}"
    gt_sig = pilimg.open(gt_data_dir)
    gt_sig = np.array(gt_sig)
    gt_data = gt_sig[test_result[:, 0], test_result[:, 1]] - 1
    confusion_mat = confusion_matrix(test_result[:, 2], gt_data)

    return confusion_mat

def calculate_score(confusion_mat):
    W_precision = confusion_mat[0, 0] / confusion_mat[0].sum()
    W_recall = confusion_mat[0, 0] / confusion_mat[:, 0].sum()
    W_F1 = (2 * W_precision * W_recall) / (W_precision + W_recall)

    V_W_precision = confusion_mat[1, 1] / confusion_mat[1].sum()
    V_W_recall = confusion_mat[1, 1] / confusion_mat[:, 1].sum()
    V_W_F1 = (2 * V_W_precision * V_W_recall) / (V_W_precision + V_W_recall)

    Se2_precision = confusion_mat[2, 2] / confusion_mat[2].sum()
    Se2_recall = confusion_mat[2, 2] / confusion_mat[:, 2].sum()
    Se2_F1 = (2 * Se2_precision * Se2_recall) / (Se2_precision + Se2_recall)

    Vac_Se_precision = confusion_mat[3, 3] / confusion_mat[3].sum()
    Vac_Se_recall = confusion_mat[3, 3] / confusion_mat[:, 3].sum()
    Vac_Se_F1 = (2 * Vac_Se_precision * Vac_Se_recall) / (Vac_Se_precision + Vac_Se_recall)

    Vac_Se2_precision = confusion_mat[4, 4] / confusion_mat[4].sum()
    Vac_Se2_recall = confusion_mat[4, 4] / confusion_mat[:, 4].sum()
    Vac_Se2_F1 = (2 * Vac_Se2_precision * Vac_Se2_recall) / (Vac_Se2_precision + Vac_Se2_recall)
    
    accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1] + confusion_mat[2, 2] + confusion_mat[3, 3] + confusion_mat[4, 4]) * 100 / confusion_mat.sum()

    score_result = [W_precision, W_recall, W_F1,
                    V_W_precision, V_W_recall, V_W_F1,
                    Se2_precision, Se2_recall, Se2_F1,
                    Vac_Se_precision, Vac_Se_recall, Vac_Se_F1,
                    Vac_Se2_precision, Vac_Se2_recall, Vac_Se2_F1,
                    accuracy]
    
    print(f"W Precision : {W_precision:.3f} | W Recall : {W_recall:.3f} | W F1 score : {W_F1:.3f}")
    print(f"V_W Precision : {V_W_precision:.3f} | V_W Recall : {V_W_recall:.3f} | V_W F1 score : {V_W_F1:.3f}")
    print(f"Se2 Precision : {Se2_precision:.3f} | Se2 Recall : {Se2_recall:.3f} | Se2 F1 score : {Se2_F1:.3f}")
    print(f"Vac_Se Precision : {Vac_Se_precision:.3f} | Vac_Se Recall : {Vac_Se_recall:.3f} | Vac_Se F1 score : {Vac_Se_F1:.3f}")
    print(f"Vac_Se2 Precision : {Vac_Se2_precision:.3f} | Vac_Se2 Recall : {Vac_Se2_recall:.3f} | Vac_Se2 F1 score : {Vac_Se2_F1:.3f}")
    print(f"Accuracy : {accuracy:.3f} %")

    return score_result


def get_result_df():

    result_df = pd.DataFrame({"data_dir" : [],
                            "W counts" : [],
                            "V_W counts" : [],
                            "Se2 counts" : [],
                            "Vac_Se counts" : [],
                            "Vac_Se2 counts" : [],
                            "V_dopants (%)" : [],
                            "Se_vacancies (%)" : []
                            })
    
    return result_df

def update_result_df(result_df, test_data_dir, test_result):
    df_idx = len(result_df)
    W_counts = (test_result[:, 2] == 0).sum()
    V_W_counts = (test_result[:, 2] == 1).sum()
    Se2_counts = (test_result[:, 2] == 2).sum()
    Vac_Se_counts = (test_result[:, 2] == 3).sum()
    Vac_Se2_counts = (test_result[:, 2] == 4).sum()
    V_dopants = V_W_counts * 100 / (W_counts + V_W_counts + 2 * (Se2_counts + Vac_Se_counts + Vac_Se2_counts))
    Se_vacancies = (Vac_Se_counts + Vac_Se2_counts * 2) * 100 / (W_counts + V_W_counts + 2 * (Se2_counts + Vac_Se_counts + Vac_Se2_counts))
    result_df.loc[df_idx, "data_dir"] = test_data_dir
    result_df.loc[df_idx, "W counts"] = int(W_counts)
    result_df.loc[df_idx, "V_W counts"] = int(V_W_counts)
    result_df.loc[df_idx, "Se2 counts"] = int(Se2_counts)
    result_df.loc[df_idx, "Vac_Se counts"] = int(Vac_Se_counts)
    result_df.loc[df_idx, "Vac_Se2 counts"] = int(Vac_Se2_counts)
    result_df.loc[df_idx, "V_dopants (%)"] = V_dopants
    result_df.loc[df_idx, "Se_vacancies (%)"] = Se_vacancies

    return result_df


