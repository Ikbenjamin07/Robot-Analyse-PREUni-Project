import numpy as np
from PIL import Image
import os

def main():
    SF()

def SF():
    # Define the path to the folder
    folder_path = 'single_robot/robots/'
    spatial_frequencies = []
    # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        image = Image.open(file_path).convert('L')
        image_data = np.array(image)

        # Calculate Row Frequency (RF)
        RF = np.sqrt(np.sum(np.diff(image_data, axis=0)**2) / (image_data.shape[0] * image_data.shape[1]))

        # Calculate Column Frequency (CF)
        CF = np.sqrt(np.sum(np.diff(image_data, axis=1)**2) / (image_data.shape[0] * image_data.shape[1]))

        # Calculate Spatial Frequency (SF)
        SF = np.sqrt(RF**2 + CF**2)
        spatial_frequencies.append(SF)



    return spatial_frequencies

if __name__ == "__main__":
    main()