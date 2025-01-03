from data_provider.data_loader import Dataset_T1DM, Dataset_ean, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
    'ean': Dataset_ean,
    'T1DM': Dataset_T1DM,
}


def data_provider(args, flag):
    """
    Loads the dataset based on the provided arguments and phase (train/val/test).

    :param args: Command-line arguments.
    :param flag: 'train', 'val', or 'test'.
    :return: Dataset and DataLoader for the specified phase.
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent
    
     # Determine data file path
    if flag == 'train' and args.train_data_path:
        data_path = args.train_data_path
    elif flag == 'val' and args.train_data_path:
        data_path = args.train_data_path
    elif flag == 'test' and args.test_data_path:
        data_path = args.test_data_path
    else:
        data_path = args.data_path

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
