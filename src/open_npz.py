from numpy import load

data = load('src/npz_e65_data/E65_data.npz')
lst = data.files 
for item in lst[:100]:
    print(f"{item}: shape={data[item].shape}, dtype={data[item].dtype}")