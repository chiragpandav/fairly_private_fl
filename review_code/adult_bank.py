import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
sys.path.append("..")
from utils import to_numeric
import pickle


# this has changed for Bank data

class ADULT(BaseDataset):

    def __init__(self, name='ADULT', single_bit_binary=False, device='cpu', random_state=42,name_state=0):
        super(ADULT, self).__init__(name=name, device=device, random_state=random_state)

        self.features = {
            'age': None,
            'job': ['management', 'technician', 'entrepreneur', 'retired', 'admin.',
                   'services', 'blue-collar', 'self-employed', 'unemployed',
                   'housemaid', 'student'],
            'education': ['tertiary', 'secondary', 'primary'],
            'default':['no', 'yes'],
            'balance':None ,
            'housing': ['no', 'yes'],
            'loan': ['no', 'yes'],
            'day_of_week': None,
            'marital': ['married', 'single', 'divorced'],
            'age_group': None,            
            
            'month': ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
                       'mar', 'apr', 'sep'],
            'duration': None,
            'campaign': None,            
            'previous':None,
            'y': ['no', 'yes']
        }

        self.single_bit_binary = single_bit_binary
        self.label = 'y'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}
        print("State Code:: ",name_state)

        train_data_df = pd.read_csv(f'clients_data/raw_data/client_{name_state}.data', delimiter=',', names=list(self.features.keys()), engine='python')
        test_data_df = pd.read_csv(f'clients_data/raw_data/client_bank.test', delimiter=',', names=list(self.features.keys()), skiprows=1, engine='python')


        # if uncomment,  self.features order needed to be changed
        # train_data_df = pd.read_csv('datasets/ADULT/bank.data', delimiter=',', names=list(self.features.keys()), engine='python')
        # test_data_df = pd.read_csv('datasets/ADULT/bank.test', delimiter=',', names=list(self.features.keys()), skiprows=1, engine='python')

        train_data = train_data_df.to_numpy()
        test_data = test_data_df.to_numpy()

        # drop missing values
        # note that the category never worked always comes with a missing value for the occupation field, hence this
        # step effectively removes the never worked category from the dataset
        train_rows_to_keep = [not ('?' in row) for row in train_data]
        test_rows_to_keep = [not ('?' in row) for row in test_data]
        train_data = train_data[train_rows_to_keep]
        test_data = test_data[test_rows_to_keep]

        # remove the annoying dot from the test labels
        # for row in test_data:
        #     row[-1] = row[-1][:-1]

        # convert to numeric features
        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)

        # split features and labels
        Xtrain, Xtest = train_data_num[:, :-1].astype(np.float32), test_data_num[:, :-1].astype(np.float32)
        ytrain, ytest = train_data_num[:, -1].astype(np.float32), test_data_num[:, -1].astype(np.float32)
        self.num_features = Xtrain.shape[1]
        
        print(np.unique(ytrain))
        print(np.unique(ytest))

        # transfer to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()

    def load_gmm(self, base_path):
        # TODO: this functionality has to be extended to all classes implementing Base_dataset
        with open(base_path + '/ADULT/fitted_gmms/all_cont_gmm.sav', 'rb') as f:
            gmm = pickle.load(f)
        self.gmm_parameters = {
            'all': (torch.as_tensor(gmm.weights_, device=self.device),
                    torch.as_tensor(gmm.means_, device=self.device),
                    torch.as_tensor(gmm.covariances_, device=self.device))
        }
        for feature_name, (feature_type, _) in self.train_feature_index_map.items():
            if feature_type == 'cont':
                with open(base_path + f'/ADULT/fitted_gmms/{feature_name}_gmm.sav', 'rb') as f:
                    gmm = pickle.load(f)
                self.gmm_parameters[feature_name] = (torch.as_tensor(gmm.weights_, device=self.device),
                                                     torch.as_tensor(gmm.means_, device=self.device),
                                                     torch.as_tensor(gmm.covariances_, device=self.device))
        self.gmm_parameters_loaded = True
