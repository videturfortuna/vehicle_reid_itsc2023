import copy
import random
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import torchvision


def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids 

        
class CustomDataSet4VERIWILD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = self.data_info.iloc[idx, 2]
        
        view_id = 0 #self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view :
            return img, vid, camid, view_id
        else:
            return img, vid, camid, 0





class CustomDataSet4VERIWILDv2(Dataset):
    """VeriWild 2.0 dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = 0 #self.data_info.iloc[idx, 2]
        view_id = 0 # = self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view:
            return img, vid, camid, view_id
        else:
            return img, vid, camid



class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

        
class CustomDataSet4Market1501(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])  
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].split('s')[0].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid     

       
 


class CustomDataSet4Veri776(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])     
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
                # print(pid, id, 'debug')
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0 






class CustomDataSet4Veri776_withviewpont(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, viewpoint_train, viewpoint_test, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.viewpoint_train = pd.read_csv(viewpoint_train, sep=' ', header = None)
        self.viewpoint_test = pd.read_csv(viewpoint_test, sep=' ', header = None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        self.view = []
        conta_missing_images = 0
        if is_train == True:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line]
                if self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line].shape[0] ==0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1]) 
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line]
                if self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line].shape[0] == 0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform
        print('Missed viewpoint for ', conta_missing_images, ' images!')
    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))-1
        viewid = np.int64(self.view[idx])


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, viewid     

class CustomDataSet4VehicleID_Random(Dataset):
    def __init__(self, lines, root_dir, is_train=True, mode=None, transform=None, teste=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = []
        self.names = []
        self.labels = []
        self.teste = teste
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        ### no camera information
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.teste:
            return img, vid, camid, 0
        else:
            return img, vid, camid





class CustomDataSet4VehicleID(Dataset):
    def __init__(self, image_list, root_dir, is_train=True, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0


