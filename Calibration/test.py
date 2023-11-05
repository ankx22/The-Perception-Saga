import pickle

# Open the file in binary read mode
with open('dist.pkl', 'rb') as file:
    # Load the contents from the file and deserialize it
    calibration_matrix = pickle.load(file)

# Now you can use the calibration_matrix object
print(calibration_matrix)