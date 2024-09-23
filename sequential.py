import itertools

import pandas as pd

# Load the general.csv file
df = pd.read_csv("/home/fanis/code/python-db/export/general.csv")

# Replace the ratings 1 and 2 with 'positive' and -1 and -2 with 'negative'
df["rating"] = df["rating"].replace(
    {1: "positive", 2: "positive", -1: "negative", -2: "negative"}
)

# Sort the DataFrame
df = df.sort_values("id")

# Initialize the maximum positive and negative lengths and the corresponding starting and ending ids
max_pos_length = max_neg_length = 0
start_pos_id = end_pos_id = start_neg_id = end_neg_id = None

# Initialize the total length and the number of groups
total_length = num_groups = 0

# Group by 'rating' and iterate over the groups
for rating, group in itertools.groupby(df.itertuples(), lambda x: x.rating):
    # Convert the group to a list
    group = list(group)

    # If the rating is sequential
    if rating in ["positive", "negative"]:
        # Update the total length and the number of groups
        total_length += len(group)
        num_groups += 1

        # If the length of the group is greater than the maximum length
        if len(group) > max_pos_length and rating == "positive":
            # Update the maximum positive length and the corresponding starting and ending ids
            max_pos_length = len(group)
            start_pos_id = group[0].id
            end_pos_id = group[-1].id
        elif len(group) > max_neg_length and rating == "negative":
            # Update the maximum negative length and the corresponding starting and ending ids
            max_neg_length = len(group)
            start_neg_id = group[0].id
            end_neg_id = group[-1].id

# Calculate the average length of the sequential ratings
average_length = total_length / num_groups

# Create a new DataFrame with the maximum sequential ratings and save it to a CSV file
max_ratings_df = pd.DataFrame(
    {
        "rating": ["positive", "negative"],
        "length": [max_pos_length, max_neg_length],
        "start_id": [start_pos_id, start_neg_id],
        "end_id": [end_pos_id, end_neg_id],
    }
)
max_ratings_df.to_csv("/home/fanis/code/python-db/export/max_ratings.csv", index=False)

# Create another DataFrame with the average length and save it to another CSV file
average_length_df = pd.DataFrame({"average_length": [average_length]})
average_length_df.to_csv(
    "/home/fanis/code/python-db/export/average_length.csv", index=False
)
