import pandas as pd 

df_40 = pd.read_csv('/media/sonnh/kaggle-vin/runs/train/train15_fold2/ap_40_per_class.csv')
df_50 = pd.read_csv('/media/sonnh/kaggle-vin/runs/train/train15_fold2/ap_50_per_class.csv')

names = ['Aortic enlargement',
        'Atelectasis',
        'Calcification',
        'Cardiomegaly',
        'Consolidation',
        'ILD',
        'Infiltration',
        'Lung Opacity',
        'Nodule/Mass',
        'Other lesion',
        'Pleural effusion',
        'Pleural thickening',
        'Pneumothorax',
        'Pulmonary fibrosis']

max_epoch_40 = list(df_40.idxmax(axis = 0))[2:]
max_epoch_50 = list(df_50.idxmax(axis = 0))[2:]
max_ap_40 = []
max_ap_50 = []
for i in range(14):
    max_ap_40.append(df_40[str(i)][max_epoch_40[i]])
    max_ap_50.append(df_50[str(i)][max_epoch_50[i]])

result = {'class':names, 'max_ap40':max_ap_40, 'ap_40_epoch':max_epoch_40,  'max_ap50':max_ap_50, 'ap_50_epoch':max_epoch_50}

pd.DataFrame(result).to_csv('/media/sonnh/kaggle-vin/runs/train/train15_fold2/summary.csv', index = False)
    



