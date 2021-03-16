import pandas as pd

def summary_res_ap_40(save_dir):
    df = pd.read_csv('{}/ap_40_per_class_per_rad.csv'.format(save_dir))


    res_csv_40 = {'epoch': df['epoch'].tolist()}
    for name in range(14):
        for r in ['val_r8', 'val_r9', 'val_r10']:
            if name not in res_csv_40:
                res_csv_40[name] = df['{}_{}'.format(name, r)]
            else:
                res_csv_40[name] += df['{}_{}'.format(name, r)]
    res_csv_40= pd.DataFrame(res_csv_40)
    res_csv_40.head()

    for name in range(14):
        res_csv_40[name]/=3
    res_csv_40.head()

    res_csv_40["ap"] = res_csv_40[range(14)].sum(axis=1) /14


    res_csv_40.to_csv('{}/ap_40_per_class_per_rad_summary.csv'.format(save_dir))
