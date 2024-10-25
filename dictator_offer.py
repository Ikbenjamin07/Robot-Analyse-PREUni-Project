import csv
import numpy as np
import os
import pandas as pd

def main():
    Get_Lists()

def Get_Lists():
    # Open the CSV file and read the specified column
    with open("indian_data.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column_data = [float(row['dictator_offer']) for row in reader]  # Replace 'column_name' with your column name

    # Convert to a NumPy matrix (as a column vector)
    robots = []
    folder_path = 'img/'
        # Loop over each file in the folder
    for filename in os.listdir(folder_path):
        robots.append(filename[:-4])


    tempValues = []
    for i in range (0, 18):
        tempValues.append([])


    for i, e in enumerate(column_data):
        tempValues[i%18].append(e)  
    robotsValues = []
    for robot in tempValues:
        total = 0
        for i in robot:
            total += i
        robotsValues.append(total/len(robot))
    
    return robots, robotsValues


if __name__ == "__main__":
    main()