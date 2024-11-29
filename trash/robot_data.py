import pandas as pd
import os

def main():
    print(get_dictator_offer_per_robot())


def get_dictator_offer_per_robot():
    dictator_offer_data = get_column_data("indian_data.csv", "dictator_offer")
    list_robots = make_robot_list("img/")
    mean_offer_robot = mean_dictator_offer_robot(dictator_offer_data)
    list_of_robots = list_robots

    df = pd.DataFrame({
        "Robot_name": list_of_robots,
        "mean_dictator_offer": mean_offer_robot
    })

    return df

def get_column_data(csv_file, column_name):
    df = pd.read_csv(csv_file)
    if column_name in df.columns:
        return df[column_name].tolist()

def mean_dictator_offer_robot(dictator_offers):
    mean_offer_list = []
    for i in range(18):
        total_offer_value = 0
        for j in range(len(dictator_offers)):
            if j % 18 == i:
                total_offer_value = total_offer_value + dictator_offers[j]
        
        mean_offer_list.append(total_offer_value/(len(dictator_offers)/18))
    return mean_offer_list

def make_robot_list(directory):
    # List to store image file names
    image_files = []

    # Loop through the files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file ends with .PNG or .jpg
        if filename.lower().endswith(('.png', '.jpg')):
            image_files.append(filename[:-4])
    image_files.sort()
    return image_files

if __name__ == "__main__":
    main()