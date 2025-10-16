import h5py

f = h5py.File("../E65_struct.mat", "r")
print(f["/nic_output/behavioralVariables"].keys())