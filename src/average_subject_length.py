import os
import pandas as pd


class SubjectAnalyzer:
    def __init__(self):
        self.current_dir = os.getcwd()
        self.export_dir = os.path.join(self.current_dir, "export")
        self.data_path = os.path.join(self.export_dir, "general.csv")

    def analyze_subject_lengths(self):
        df = pd.read_csv(self.data_path)
        df["subject_length"] = df.groupby("subject")["id"].transform("nunique")
        average_length = df["subject_length"].mean()

        pd.DataFrame({"average_length": [average_length]}).to_csv(
            os.path.join(self.export_dir, "average_length.csv"),
            index=False
        )

    def analyze_streaks(self, min_streak_length=10):
        df = pd.read_csv(self.data_path)
        streak = 0
        streak_start = 0
        streak_type = None
        subjects = set()
        results = []

        for i, row in df.iterrows():
            rating = row["rating"]
            subject = row["subject"]

            if (rating > 0 and streak_type == "positive") or (
                rating < 0 and streak_type == "negative"
            ):
                streak += 1
                subjects.add(subject)
            else:
                if streak > min_streak_length:
                    results.append(
                        (streak_start, i - 1, streak_type, streak, len(subjects)))
                streak = 1
                streak_start = i
                streak_type = "positive" if rating > 0 else "negative"
                subjects = {subject}

        if streak > min_streak_length:
            results.append(
                (streak_start, i, streak_type, streak, len(subjects)))

        pd.DataFrame(
            results,
            columns=["start", "end", "type", "length", "num_subjects"]
        ).to_csv(os.path.join(self.export_dir, "streaks_more_than_10.csv"), index=False)

    def run_analysis(self):
        self.analyze_subject_lengths()
        self.analyze_streaks()


if __name__ == "__main__":
    analyzer = SubjectAnalyzer()
    analyzer.run_analysis()
