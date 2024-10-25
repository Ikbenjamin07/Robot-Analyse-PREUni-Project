import csv
import numpy as np

# Open the CSV file and read the specified column
with open("indian_data.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    column_data = [float(row['dictator_offer']) for row in reader]  # Replace 'column_name' with your column name

# Convert to a NumPy matrix (as a column vector)

robots = []
for i in range (0, 18):
    robots.append([])


for i, e in enumerate(column_data):
    robots[i%18].append(e)  

robotsValues = []

for robot in robots:
    total = 0
    for i in robot:
        total += i
    robotsValues.append(total/len(robot))



