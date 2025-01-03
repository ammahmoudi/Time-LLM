import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
import warnings

warnings.filterwarnings('ignore')

class Dataset_T1DM(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='train.csv',
                 target='_value', scale=True, timeenc=0, freq='5min', percent=100,
                 seasonal_patterns=None, scaler=None):
        """
        Dataset for T1DM glucose levels.

        :param root_path: Root directory containing the dataset files.
        :param flag: 'train', 'val', or 'test'.
        :param size: Tuple of (seq_len, label_len, pred_len).
        :param features: 'S' for single target or 'M' for multiple features.
        :param data_path: Path to the dataset file (train or test file).
        :param target: Target column name (default is '_value').
        :param scale: Whether to scale the data (default is True).
        :param timeenc: Whether to encode time features (0 for no, 1 for yes).
        :param freq: Frequency of the time series (default is '5min').
        :param percent: Percentage of training data (default is 100%).
        :param scaler: external scaler ( used to pass thet train dataset fitted scaler to the test dataset)
        """
        if size is None:
            self.seq_len = 12  # Default: 12 samples (60 minutes if every 5 mins)
            self.label_len = 6  # Default: 6 samples (30 minutes)
            self.pred_len = 12  # Default: Predict 12 samples (60 minutes)
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.root_path = root_path
        self.data_path = os.path.join(root_path, data_path)
        self.features=features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.seasonal_patterns = seasonal_patterns
        self.scaler = scaler 
        self._data_transformed = False  # Initialize the flag for scaling status

        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]  # Number of input features
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1  # Total valid sequences

    def __read_data__(self):
        """
        Reads and preprocesses the data.
        """
        # Load the data
        df_raw = pd.read_csv(self.data_path)

        # Ensure correct columns exist
        assert '_ts' in df_raw.columns and self.target in df_raw.columns, \
            "Dataset must contain '_ts' (timestamp) and target columns."

        # Sort by timestamp
        df_raw['_ts'] = pd.to_datetime(df_raw['_ts'])
        df_raw = df_raw.sort_values('_ts')
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            
        # Extract values
        data = df_data.values
        
        # Split data for train and val modes
        num_samples = len(data)
        num_train = int(num_samples * 0.7)

        if self.flag == 'train':
            border1, border2 = 0, num_train
        elif self.flag == 'val':
            border1, border2 = num_train, num_samples
        else:  # Test mode
            border1, border2 = 0, num_samples

        data = data[border1:border2]
        
         # Use external scaler if provided, else fit during training (if use differnet x and y fix this to scale properly)
        if self.scale:
             if self.flag == 'train' and self.scaler is None:
             # Fit a new scaler during training
                self.scaler = StandardScaler()
                self.scaler.fit(data)
                data = self.scaler.transform(data)
                self._data_transformed = True
                
             elif self.scaler:  # Use the provided scaler
                data = self.scaler.transform(data)
                self._data_transformed = True
             elif self.flag in ['val', 'test']:
                # Defer scaling if scaler is not yet provided
                self._raw_data = data
                self._data_transformed = False
            
         # Process time features
        df_stamp = df_raw.iloc[border1:border2][['_ts']]
        if self.timeenc == 0:
            # Manually extract time-related features
            df_stamp['month'] = df_stamp['_ts'].dt.month
            df_stamp['day'] = df_stamp['_ts'].dt.day
            df_stamp['weekday'] = df_stamp['_ts'].dt.weekday
            df_stamp['hour'] = df_stamp['_ts'].dt.hour
            df_stamp['minute'] = df_stamp['_ts'].dt.minute // (60 // 12)  # Convert minutes into bins (5-min intervals)
            self.data_stamp = df_stamp[['month', 'day', 'weekday', 'hour', 'minute']].values
        elif self.timeenc == 1:
            # Use a learned encoding for time features
            self.data_stamp = time_features(pd.to_datetime(df_stamp['_ts'].values), freq=self.freq)
            self.data_stamp = self.data_stamp.transpose(1, 0)
       
        self.data_x = data
        self.data_y = data
        

    def __getitem__(self, index):
        """
        Returns the input sequence, target sequence, and time features.

        :param index: Index of the starting position for sequence generation.
        :return: Tuple of (input_sequence, target_sequence, time_features_input, time_features_target).
        """
        feat_id = 0  # Default to the first feature (univariate)
        if self.features == 'M':  # For multivariate input
            feat_id = slice(None)  # Select all features

        seq_x = self.data_x[index:index + self.seq_len, feat_id]
        seq_y = self.data_y[index + self.seq_len:index + self.seq_len + self.pred_len, feat_id]
        seq_x_mark = self.data_stamp[index:index + self.seq_len]
        seq_y_mark = self.data_stamp[index + self.seq_len:index + self.seq_len + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Returns the number of available samples in the dataset.
        """
        return self.tot_len

    def inverse_transform(self, data):
        """
        Reverses the scaling transformation for interpretability.

        :param data: Scaled data.
        :return: Original data in the original scale.
        """
        if self.scale:
            return self.scaler.inverse_transform(data)
        else:
            return data
    def set_scaler(self, scaler):
        """
        Set an external scaler for validation or test datasets and transform the data if not already scaled.

        :param scaler: Pre-fitted scaler (e.g., from the training data).
        """
        self.scaler = scaler
        if self.scaler and self.scale:
            # Ensure the data is transformed only if it hasn't already been scaled
            if not hasattr(self, '_data_transformed') or not self._data_transformed:
                self.data_x = self.scaler.transform(self.data_x)
                self.data_y = self.scaler.transform(self.data_y)
                self._data_transformed = True  # Mark that the data has been transformed




class Dataset_ean(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='data.csv',
                 target='sold_units', scale=False, timeenc=0, freq='W', percent=100,
                 seasonal_patterns=None):
        if size is None:
            self.seq_len = 13  # Use 0.25 year of data for sequence
            self.label_len = 4  # Labels from last month
            self.pred_len = 4   # Predict 1 month ahead
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw.drop("Unnamed: 0", axis=1, inplace=True)# don't forget to remove this once regulated the data
        num_weeks = len(df_raw)
        num_train = int(num_weeks * 0.7)
        num_val = int(num_weeks * 0.2)
        num_test = num_weeks - num_train - num_val

        border1s = [0, num_train, num_train + num_val]
        border2s = [num_train, num_train + num_val, num_weeks]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_raw.iloc[:num_train]
            self.scaler.fit(train_data[[self.target]].values)
            data = self.scaler.transform(df_raw[[self.target]].values)
        else:
            data = df_raw[[self.target]].values

        df_stamp = pd.to_datetime(df_raw.iloc[:, 0][border1:border2])  # 'end_date' is first column
        
        time_features = np.vstack((df_stamp.dt.year, df_stamp.dt.month, df_stamp.dt.day, df_stamp.dt.weekday)).T
       

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.time_features = time_features

    

    def __getitem__(self, index):
        seq_x = self.data_x[index:index+self.seq_len]
        seq_y = self.data_y[index+self.seq_len:index+self.seq_len+self.pred_len]
        seq_x_mark = self.time_features[index:index+self.seq_len]
        seq_y_mark = self.time_features[index+self.seq_len:index+self.seq_len+self.pred_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask

