import pandas as pd

# Load the CSV file
df = pd.read_csv("/home/fanis/code/python-db/export/grouped_data.csv")

# Sort the DataFrame

# Group by 'group_id' and calculate the length of each group
df["group_length"] = df.groupby("group")["id"].transform("nunique")

# Calculate the average length
average_length = df["group_length"].mean()

print(f"The average length of each group is {average_length}")
