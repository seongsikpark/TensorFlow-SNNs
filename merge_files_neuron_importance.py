

import pandas as pd

print('start')

#save_dir = './result_amap_cond_n_scnorm'
save_dir = './result_amap_cond_n_scnorm_pg'
mode='NORMAL'
mode='WTA-1'
mode='SIM-A'
mode='SIM-S'

f_name_pre=save_dir+'/'+mode

dfs = []
for batch_idx in range(0,100):
    f_name=f_name_pre+'_b-'+str(batch_idx)+'.xlsx'
    df = pd.read_excel(f_name)
    dfs.append(df)

cdf = pd.concat(dfs, ignore_index=True)

cdf.drop(df.columns[0],axis=1,inplace=True)

cdf.loc['mean']=cdf.mean()

f_name_mg = f_name_pre+'_mg.xlsx'

cdf.to_excel(f_name_mg)


print('output file: '+f_name_mg)
print('end')