from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Synthetic, Dataset_Tabular
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    # 3-regime synthetic series from scripts/make_synth_moe3.py. Plain Dataset_Custom
    # (same CSV layout, same 70/10/20 split, same train-fit scaling as the real
    # datasets, so the numbers are comparable); it earns its own key only so the
    # `setting` string records the dataset name instead of a generic "custom".
    'SynthMoE3': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'Synthetic': Dataset_Synthetic,
    'Bike': Dataset_Tabular,
    'Temperature': Dataset_Tabular,
    'Superconductivity': Dataset_Tabular
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or flag == 'val') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'tabular_regression':
        dataset = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            data_name=args.data,
            flag=flag,
            size=[args.n_samples] if getattr(args, 'n_samples', None) is not None else None,
            seed=getattr(args, 'seed', 42)
        )

        shuffle_flag = True if flag == 'train' else False
        drop_last = True if flag == 'train' else False
        batch_size = args.batch_size
        
        data_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle_flag, 
            num_workers=args.num_workers, 
            drop_last=drop_last
        )
        return dataset, data_loader
    elif args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
