import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Reading data from running folder
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'cicids2017_cleaned.csv')

df = pd.read_csv(file_path)

#print(sorted(df.columns.tolist()))

#erase spaces from column names
df.columns = df.columns.str.strip()

# Shuffle data and takes 100 000 row sample
# random_state=42 ensures same result every time
df_sample = shuffle(df, random_state=42).iloc[:100000, :].copy()

print(f"Otannan koko: {df_sample.shape}")

features = [
    'Bwd Packet Length Std', 'Subflow Fwd Bytes', 'Flow Duration', 
    'Total Length of Fwd Packets', 'Init_Win_bytes_forward', 'Flow IAT Std', 
    'Active Mean', 'Bwd Packets/s', 'Fwd Packet Length Mean', 'Bwd Packet Length Min'
]

target = 'Attack Type'

# Validating that all are found
missing = [f for f in features if f not in df_sample.columns]
if missing:
    print(f"VAROITUS: Seuraavia sarakkeita ei löydy: {missing}")

# Filter only these columsn
df_final = df_sample[features + [target]].copy()

# Puhdistus: Korvataan Infinity NaN-arvolla ja poistetaan ne
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final.dropna(inplace=True)

print(f"Koko puhdistuksen jälkeen: {df_final.shape}")

# Phase A: 60% training data, 40% other (validation + test)
train_df, temp_df = train_test_split(
    df_final, test_size=0.4, random_state=42, stratify=df_final[target]
)

# Phase B: Spliet leftover 40% in half -> 20% validation / 20% test
# (0.5 * 0.4 = 0.2)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df[target]
)

print(f"Opetus: {len(train_df)}, Validaatio: {len(val_df)}, Testi: {len(test_df)}")

train_path = os.path.join(current_dir, 'train_data.csv')
val_path = os.path.join(current_dir, 'val_data.csv')
test_path = os.path.join(current_dir, 'test_data.csv')

# Save files
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Tiedostot tallennettu kansioon: {current_dir}")