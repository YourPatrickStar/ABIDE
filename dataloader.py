import os
import numpy as np
import torch
import torch.utils.data as utils
import csv
import scipy.io as sio
from PIL import Image
from nilearn import connectome
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import glob
import random


phenotype = "PLSNet/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
data_folder = "PLSNet/ABIDE_pcp/cpac/filt_noglobal"


def sliding_window_sampling(timeseries, label, window_size=70, step_size=20):
    """
    使用滑动窗口从每个样本中提取多个固定长度的时间片段
    Parameters:
        data (list of np.array): 每个样本的数据，形状为 (num_roi, timepoints)。
        window_size (int): 窗口大小，默认值为70。
        step_size (int): 步长，默认为1。
    Returns:
        samples (list of np.array): 处理后的数据，每个元素的形状为 (num_roi, window_size)。
    """
    samples = []
    labels = []
    indices = []
    for j in range(len(timeseries)):
        sample = timeseries[j]
        y = label[j]
        num_roi, timepoints = sample.shape
        # 计算滑动窗口的数量
        num_windows = (timepoints - window_size) // step_size + 1
        # 对每个时间序列，截取窗口
        for i in range(num_windows):
            # 获取当前窗口的数据片段
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window = sample[:, start_idx:end_idx]
            samples.append(window)
            labels.append(y)
            indices.append(j)
    return samples, labels, indices


def random_crop_to_fixed_length(test_ts, target_length=70):
    """
    对每个样本进行随机截取，保留长度为target_length的时间序列数据
    Parameters:
        test_ts (list of np.array): 每个元素为一个样本，形状为 (num_roi, timepoints)
        target_length (int): 截取后的时间序列长度，默认为70
    Returns:
        list of np.array: 返回每个样本随机截取后的数据，形状为 (num_roi, target_length)
    """
    new_test_ts = []
    for sample in test_ts:
        num_roi, timepoints = sample.shape
        # 检查时间点长度是否大于等于target_length
        if timepoints >= target_length:
            # 随机选择一个起始点
            start_idx = random.randint(0, timepoints - target_length)
            # 截取长度为target_length的时间序列
            cropped_sample = sample[:, start_idx:start_idx + target_length]
        else:
            # 如果时间点不足70，可以选择填充或忽略，假设数据都满足条件
            raise ValueError("Sample has less than the target length. Consider padding or removing this sample.")
        new_test_ts.append(cropped_sample)
    return new_test_ts


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

#获取时间序列
def get_timeseries(subject_list, atlas_name):
    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        # print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=1).T) #111*t
        #从_rois_ho.1D文件中读取时间序列，转置后存储在timeseries列表中
    print("Reading timeseries file: done")
    return timeseries

#分割每个样本的时序列，每100个时间点分一次，返回id列表和时间序列列表
def subject_timeseries_segmentation(subject_list, atlas_name):
    subject_id = []
    timeseries_list = []
    timeseries = get_timeseries(subject_list, atlas_name)
    for id, t in zip(subject_list, timeseries):
        t2 = np.array([])
        t3 = np.array([])
        tmp = 0
        if t.shape[1] >= 300:
            t2 = t[:, 100:200]
            t3 = t[:, 200:300]
            tmp = 2
        elif t.shape[1] >= 200:
            t2 = t[:, 100:200]
            tmp = 1
        else:
            continue
        if tmp == 1:
            subject_id.append(id)
            timeseries_list.append(t2)
        if tmp == 2:
            subject_id.extend([id, id])
            timeseries_list.append(t2)
            timeseries_list.append(t3)
    subject_id = np.array(subject_id)
    timeseries_list = np.array(timeseries_list)
    return subject_id, timeseries_list

def load_seg_new(subject_list, atlas_name, kind):
    seg_dict = {}
    subject_id = []
    timeseries_list = []
    network_list = []
    timeseries = get_timeseries(subject_list, atlas_name)
    for id, t in zip(subject_list, timeseries):
        t1 = np.array([])
        t2 = np.array([])
        t3 = np.array([])
        tmp = 0
        if t.shape[1] >= 300:
            t1 = t[:, :100]
            t2 = t[:, 100:200]
            t3 = t[:, 200:300]
            tmp = 3
        elif t.shape[1] >= 200:
            t1 = t[:, :100]
            t2 = t[:, 100:200]
            tmp = 2
        elif t.shape[1] >= 100:
            t1 = t[:, :100]
            tmp = 1
        else:
            continue
        if tmp == 1:
            subject_id.append(id)
            net1 = subject_connectivity(t1, kind)
            network_list.append([net1])
            timeseries_list.append([t1])
        if tmp == 2:
            subject_id.append(id, id)
            net1 = subject_connectivity(t1, kind)
            net2 = subject_connectivity(t2, kind)

            network_list.append([net1,net2])
            timeseries_list.append([t1,t2])
        if tmp == 3:
            subject_id.append(id, id, id)
            net1 = subject_connectivity(t1, kind)
            net2 = subject_connectivity(t2, kind)
            net3 = subject_connectivity(t3, kind)
            network_list.append([net1, net2, net3])
            timeseries_list.append([t1, t2, t3])
        seg_dict["id"] = subject_id
        seg_dict["timeseries"] = timeseries_list
        seg_dict["networks"] = network_list
        label_dict = get_subject_score(subject_list, "DX_GROUP")
        labels = get_labels(subject_id, label_dict)
        labels = labels.astype(int) - 1
        seg_dict["labels"] = labels
    return seg_dict

#获取ID
def get_ids(num_subjects=None):

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

def get_connectivity(subject_list, atlas, kind="correlation", variable='connectivity'):
    networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject, subject + "_" + atlas + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]  # 加载matlab文件中的connectivity
        networks.append(matrix)
    return networks #871*111*111

#获取csv文件信息
def get_subject_score(subject_list, score): #获取每个样本的特定属性值，score代表特定的属性
    scores_dict = {}
    all_scores = []
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file) #逐行读取 CSV 文件的内容，每一行都是一个字典
        for row in reader: #遍历reader中的每一个元素，即每一个字典，即csv文件中的每一行
            if row['SUB_ID'] in subject_list:
                if score in  ["FIQ",'VIQ','PIQ']:
                    try:
                        value = float(row[score])
                    except ValueError:
                        value = np.nan
                    if value>30 and value<164:
                        all_scores.append(value)
                        scores_dict[row['SUB_ID']] = value
                    else:
                        scores_dict[row['SUB_ID']] = 95
                else:
                     scores_dict[row['SUB_ID']] = row[score]
    return scores_dict

def get_labels(subject_ids, label_dict):
    labels = []
    for i in subject_ids:
        labels.append(label_dict[i])
    return np.array(labels)

def subject_connectivity(timeseries, kind):
    networks = []
    if kind in ['tangent', 'partial correlation', 'correlation']:
        for sample in timeseries:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform([sample.T])[0]
            networks.append(connectivity)
    return networks

def load_seg(atlas_name, num_subject, kind):

    subject_list = get_ids(num_subject)
    subject_ids, timeseries_list = subject_timeseries_segmentation(subject_list, atlas_name)
    networks = []
    for id, t in zip(subject_ids, timeseries_list):
        network = subject_connectivity(t, kind)
        networks.append(network)
    label_dict = get_subject_score(subject_list, "DX_GROUP")
    labels = get_labels(subject_ids,label_dict)
    labels = labels.astype(int) - 1

    return timeseries_list, np.array(networks), labels


def load_origin(atlas_name, num_subject, kind):
    subject_list = get_ids(num_subject)
    ts = get_timeseries(subject_list, atlas_name)
    lbs = np.array(list(get_subject_score(subject_list, score='DX_GROUP').values()))
    lbs = lbs.astype(int)
    lbs = lbs - 1
    return ts, lbs

    nks = get_connectivity(subject_list, atlas_name, kind)
    # 筛选出时间序列小于100的样本，并将所有时间序列缩减到100
    timeseries = []

    networks = []
    labels = []
    for i in range(len(subject_list)):
        if ts[i].shape[1] < 100:
            continue
        timeseries.append(ts[i][:, :100])
        networks.append(nks[i])
        labels.append(int(lbs[i]) - 1)
    return np.array(timeseries), np.array(networks), np.array(labels)

def load_generate(kind):
    timeseries_g = []
    labels_g = []
    networks_g = []
    gen_path = "gen_samples/origin"
    for i in [0,1]:
        path = gen_path+"/samples_"+str(i)
        png_files = glob.glob(os.path.join(path, '*.png'))
        for p in png_files:
            img = Image.open(p)
            img = np.array(img).astype(np.float32) / 255.0  # Convert to float and normalize
            img = img.reshape((112, 100))[:111, :]
            timeseries_g.append(img)
            labels_g.append(i)
    for t in timeseries_g:
        network = subject_connectivity(t, kind)
        networks_g.append(network)

    return np.array(timeseries_g), np.array(networks_g), np.array(labels_g)


def load_diff_tim(dataset_config):
    atlas_name = dataset_config["atlas"]
    num_subject = dataset_config["num_subject"]
    kind = dataset_config["kind"]
    load = dataset_config["load"]
    if load == "seg":
        timeseries, _, labels = load_seg(atlas_name, num_subject, kind)
    if load == "origin":
        timeseries, _, labels = load_origin(atlas_name, num_subject, kind)
    timeseries_0 = []
    labels_0 = []
    timeseries_1 = []
    labels_1 = []

    scaler = MinMaxScaler()  # 创建 MinMaxScaler 实例

    for i in range(len(labels)):
        l = labels[i]
        ts = timeseries[i]

        # 归一化
        ts_normalized = scaler.fit_transform(ts.reshape(-1, 1)).reshape(ts.shape) #全局归一化到[0,1]
        if l == 0:
            labels_0.append(l)
            timeseries_0.append(ts_normalized)  # 使用归一化后的数据
        else:
            labels_1.append(l)
            timeseries_1.append(ts_normalized)  # 使用归一化后的数据

    return np.array(timeseries_0), np.array(labels_0), np.array(timeseries_1), np.array(labels_1)


def init_dataloader(dataset_config):
    atlas_name = dataset_config["atlas"]
    num_subject = dataset_config["num_subject"]
    kind = dataset_config["kind"]
    load = dataset_config["load"]
    len = dataset_config['timeseries_length']
    stride = dataset_config['stride']

    if load =="origin":
        # final_fc, final_pearson, labels = load_origin(atlas_name, num_subject, kind) #ndarry（846*111*100，846*111*111，846*1）
        ts, lbs = load_origin(atlas_name, num_subject, kind)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(ts):
        fold += 1
        train_ts, test_ts = [ts[i] for i in train_index], [ts[i] for i in test_index]
        train_label, test_label = [lbs[i] for i in train_index], [lbs[i] for i in test_index]
        #准备训练数据
        train_fc, train_label, _ = sliding_window_sampling(train_ts, train_label, len, stride)
        train_pearson = subject_connectivity(train_fc, kind)
        train_fc = np.array(train_fc)
        train_pearson = np.array(train_pearson)
        scaler = StandardScaler(mean=np.mean(train_fc), std=np.std(train_fc))  # 数据标准化（可以尝试z-score标准化，看有什么区别）
        train_fc = scaler.transform(train_fc)
        pseudo = []
        for i in range(train_fc.shape[0]):
            pseudo.append(np.diag(np.ones(train_pearson.shape[1])))
        train_pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))
        _, node_size, node_feature_size = train_pearson.shape
        #准备测试数据
        test_fc, labels, indices = sliding_window_sampling(test_ts, test_label, len, stride)
        indices = torch.from_numpy(np.array(indices))
        labels = torch.from_numpy(np.array(labels))
        # test_fc = random_crop_to_fixed_length(test_ts, len)
        test_pearson = subject_connectivity(test_fc, kind)
        test_fc = np.array(test_fc)
        test_pearson = np.array(test_pearson)
        scaler = StandardScaler(mean=np.mean(test_fc), std=np.std(test_fc))  # 数据标准化（可以尝试z-score标准化，看有什么区别）
        test_fc = scaler.transform(test_fc)
        pseudo = []
        for i in range(test_fc.shape[0]):
            pseudo.append(np.diag(np.ones(test_pearson.shape[1])))
        test_pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))
        train_fc, test_fc, train_pearson, test_pearson, train_pseudo, test_pseudo, train_label, test_label = \
            [torch.from_numpy(np.array(data)).float() for data in
             (train_fc, test_fc, train_pearson, test_pearson, train_pseudo_arr, test_pseudo_arr, train_label, test_label)]
        train_dataset = torch.utils.data.TensorDataset(train_fc, train_pearson, train_label, train_pseudo)
        # test_dataset = torch.utils.data.TensorDataset(test_fc, test_pearson, test_label, test_pseudo)
        test_dataset = torch.utils.data.TensorDataset(test_fc, test_pearson, labels, test_pseudo, indices)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=False)
        yield (train_dataloader, test_dataloader), node_size, node_feature_size, len, test_label

    # if load == "seg":
    #     subject_list = get_ids(num_subject)
    #     seg_dict = load_seg_new(subject_list, atlas_name, kind)
    #
    #     # final_fc, final_pearson, labels = load_origin(atlas_name, num_subject, kind)
    #     # timeseries_g, networks_g, labels_g = load_seg(atlas_name, num_subject, kind)
    #
    # if load == "gen_o":
    #     timeseries_g, networks_g, labels_g = load_generate(atlas_name, num_subject, kind)
    #     final_fc, final_pearson , labels = load_origin(atlas_name, num_subject, kind)
    # # final_fc: 所有选中样本的timeseries（num*111*100）
    # # final_person=功能连接矩阵(num*111*111)
    # # roi_num=111,
    # # pseudo=ROI位置嵌入信息(num*111*111)
    # _, _, timeseries_len = final_fc.shape  # 100
    # _, node_size, node_feature_size = final_pearson.shape #111*111
    #
    # # scaler = MinMaxScaler()# 创建 MinMaxScaler 实例
    # # final_fc_norm = []
    # # for sample in final_fc:  #对每个样本的timeseries进行全局归一化，[0，1]
    # #     ts_normalized = scaler.fit_transform(sample.reshape(-1, 1)).reshape(sample.shape)
    # #     final_fc_norm.append(ts_normalized)
    #
    # scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc)) #数据标准化（可以尝试z-score标准化，看有什么区别）
    # final_fc = scaler.transform(final_fc)
    #
    # pseudo = []
    # for i in range(len(final_fc)+len(timeseries_g) if load in ["gen_o"] else len(final_fc)):
    #     pseudo.append(np.diag(np.ones(final_pearson.shape[1]))) #样本数个对角矩阵，对角为1，其余为0，位置信息,位置one-hot编码
    #
    # if atlas_name == 'cc200':
    #     pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))
    # elif atlas_name == 'aal':
    #     pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    # elif atlas_name == 'cc400':
    #     pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    # else:
    #     pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111)) #为了与后面的功能连接矩阵进行拼接
    #
    # final_fc, final_pearson, labels, pseudo_arr = [torch.from_numpy(data).float() for data in (final_fc, final_pearson, labels, pseudo_arr)] #将这四个 NumPy 数组转换为 PyTorch 张量，并将它们的类型转换为浮点数
    # if load in ["gen_o"]:
    #     timeseries_g, networks_g, labels_g = [torch.from_numpy(data).float() for data in (timeseries_g, networks_g, labels_g)]
    #
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    #
    # fold = 0
    # for train_index, test_index in kf.split(final_fc):
    #     fold += 1
    #     # Split the data for the current fold
    #     if load in ["gen_o"]:
    #         test_fc = final_fc[test_index]
    #         test_pearson = final_pearson[test_index]
    #         test_label = labels[test_index]
    #         test_pseudo = pseudo_arr[:len(test_index)]
    #         # 训练集包含真实样本和生成的样本
    #         train_fc = final_fc[train_index]
    #         train_fc = torch.cat((train_fc, timeseries_g), dim=0)
    #         train_pearson = final_pearson[train_index]
    #         train_pearson = torch.cat((train_pearson, networks_g), dim=0)
    #         train_label = labels[train_index]
    #         train_label = torch.cat((train_label, labels_g), dim=0)
    #         train_pseudo = pseudo_arr[:len(train_index)+len(labels_g)]
    #     else:
    #         train_fc, test_fc = final_fc[train_index], final_fc[test_index]
    #         train_pearson, test_pearson = final_pearson[train_index], final_pearson[test_index]
    #         train_label, test_label = labels[train_index], labels[test_index]
    #         train_pseudo, test_pseudo = pseudo_arr[train_index], pseudo_arr[test_index]
    #
    #     # Optionally: use a validation split from train data
    #     # val_length = int(len(train_index) * dataset_config["val_set"])  # 0.1 for validation set
    #     # train_length = len(train_index) - val_length
    #     # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_length, val_length])
    #
    #     #在这里实现分割扩展
    #     # Wrap into TensorDataset
    #     train_dataset = torch.utils.data.TensorDataset(train_fc, train_pearson, train_label, train_pseudo)
    #     # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_length, val_length])
    #     test_dataset = torch.utils.data.TensorDataset(test_fc, test_pearson, test_label, test_pseudo)
    #
    #     # Create DataLoaders
    #     train_dataloader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)
    #     test_dataloader = torch.utils.data.DataLoader(
    #         test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=False)
    #     # val_dataloader = torch.utils.data.DataLoader(
    #     #     val_dataset, batch_size=dataset_config["batch_size"], shuffle=False, drop_last=False)
    #
    #
    #     # yield (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_len
    #     yield (train_dataloader, test_dataloader), node_size, node_feature_size, timeseries_len

