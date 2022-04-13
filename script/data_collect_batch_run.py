

import pandas as pd

from pandas import DataFrame

import os


data_root='../results'

#exp_name='batch_run_test_220318'
#exp_name='220320_batch_run_calib_prop'
#exp_name='220321_batch_run_calib_prop_b-200'
#exp_name='220323_batch_run_calib_prop_b-400'
#exp_name='220401_CIFAR-100_calibration_idx_test'
#exp_name='220406_vth_search_idx_test'
#exp_name='220406_vth_search_idx_test-ResNet20-CIFAR100'
#exp_name='220410_vth_search_idx_test-ResNet20-CIFAR100'
exp_name='220410_vth_search_idx_test-ResNet20-CIFAR100_ts-128'


#model_dataset='VGG16_CIFAR10'
#model_dataset='ResNet20_CIFAR100'
#model_dataset='ResNet32_CIFAR100'

path = os.path.join(data_root,exp_name)

model_datasets = os.listdir(path)

output_root = os.path.join('./',exp_name)

os.makedirs(output_root, exist_ok=True)

#
#print(model_datasets)


files = []
for model_dataset in model_datasets:

    df_acc = DataFrame()
    df_spi = DataFrame()
    path_md = os.path.join(path,model_dataset)
    files_all = os.listdir(path_md)
    files_all = sorted(files_all)
    files = [file for file in files_all if file.endswith(".xlsx")]
    for file_xlsx in files:
        file_path = os.path.join(path_md,file_xlsx)
        df_r = pd.read_excel(file_path, engine='openpyxl')
        df_r_acc = df_r['accuracy']
        df_r_spi = df_r['spike count']

        df_acc[file_xlsx] = df_r_acc
        df_spi[file_xlsx] = df_r_spi

    output_file_name_acc = 'results_collect_acc_'+model_dataset+'.xlsx'
    output_file_name_spi = 'results_collect_spi_'+model_dataset+'.xlsx'

    output_file_acc = os.path.join(output_root,output_file_name_acc)
    output_file_spi = os.path.join(output_root,output_file_name_spi)

    df_acc.to_excel(output_file_acc)
    df_spi.to_excel(output_file_spi)

print('done')
assert False

data_path = os.path.join(data_root,model_dataset)

file_name_pre = 'norm-M999_n-IF_in-REAL_nc-RATE_ts-128-10_vth-1.0_bc_cal-test-idx-'

# index
idx_s = 0
idx_e = 125

df = DataFrame()

#
for idx in range(idx_s,idx_e):
    file_name = file_name_pre + str(idx) + '.xlsx'
    file = os.path.join(data_path, file_name)
    df_r = pd.read_excel(file, engine='openpyxl')
    print(idx)
    print(df_r['accuracy'])

    df_r_acc = df_r['accuracy']

    df[idx] = df_r_acc


df = df.transpose()

df.to_excel('test_out.xlsx')
