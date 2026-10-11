import pandas as pd
import numpy as np

data = {'jumlah_kamar': [2, 3, 2, 4, 3, 5],
        'harga_juta': [400, 500, 450, 600, 550, 90000]
        }

df = pd.DataFrame(data)
print("\tRaw data")
print(df)

mean_harga = df['harga_juta'].mean()
std_harga = df['harga_juta'].std()

normal = mean_harga + (2 * std_harga)

df_bersih = df[df['harga_juta'] <= normal].copy()

# rumus standarisasi data (z score)
# z = (x - mean) / standar deviasi
mean_bersih = df_bersih['harga_juta'].mean()
std_bersih = df_bersih['harga_juta'].std()

df_bersih['harga_normal'] = (df_bersih['harga_juta'] - mean_bersih / std_bersih)

print('\n\tData bersih')
print(df_bersih)

