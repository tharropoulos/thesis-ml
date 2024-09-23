import pandas as pd

# Load the general.csv file
df = pd.read_csv("/home/fanis/code/python-db/export/general.csv")

# Group by 'subject' and calculate the number of unique ids for each subject
df["subject_length"] = df.groupby("subject")["id"].transform("nunique")

# Calculate the average length of all subjects together
average_length = df["subject_length"].mean()

# Create a new DataFrame with the average length
average_length_df = pd.DataFrame({"average_length": [average_length]})

# Save the DataFrame to a new CSV file
average_length_df.to_csv(
    "/home/fanis/code/python-db/export/average_length.csv", index=False
)


# Load the CSV file
df = pd.read_csv("/home/fanis/code/python-db/export/general.csv")

# Initialize variables
streak = 0
streak_start = 0
streak_type = None
subjects = set()
results = []

# Iterate over the rows
for i, row in df.iterrows():
    rating = row["rating"]  # replace 'rating' with your column name
    subject = row["subject"]  # replace 'subject' with your column name

    # Check if the streak continues
    if (rating > 0 and streak_type == "positive") or (
        rating < 0 and streak_type == "negative"
    ):
        streak += 1
        subjects.add(subject)
    else:
        # Check if the streak was long enough
        if streak > 10:
            results.append((streak_start, i - 1, streak_type, streak, len(subjects)))

        # Reset the streak and subjects
        streak = 1
        streak_start = i
        streak_type = "positive" if rating > 0 else "negative"
        subjects = {subject}

# Check the final streak
if streak > 10:
    results.append((streak_start, i, streak_type, streak, len(subjects)))

# Convert the results to a DataFrame and save to a CSV file
results_df = pd.DataFrame(
    results, columns=["start", "end", "type", "length", "num_subjects"]
)
results_df.to_csv("/home/fanis/code/python-db/export/streaks_more_than_10.csv")
