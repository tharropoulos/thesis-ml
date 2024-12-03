import os
import itertools
from dataclasses import dataclass
from typing import Tuple, List, Iterator
import pandas as pd


@dataclass
class RatingSequence:
    rating: str
    length: int
    start_id: int
    end_id: int


class RatingsAnalyzer:
    def __init__(self):
        self.current_dir = os.getcwd()
        self.export_dir = os.path.join(self.current_dir, "export")
        self.df = None

    def process(self) -> None:
        """Main processing pipeline."""
        self.load_and_prepare_data()
        max_sequences = self.find_max_sequences()
        avg_length = self.calculate_average_length()
        self.save_results(max_sequences, avg_length)

    def load_and_prepare_data(self) -> None:
        """Load and preprocess the data."""
        self.df = pd.read_csv(os.path.join(self.export_dir, "general.csv"))
        self.df["rating"] = self.df["rating"].replace({
            1: "positive",
            2: "positive",
            -1: "negative",
            -2: "negative"
        })
        self.df = self.df.sort_values("id")

    def find_max_sequences(self) -> Tuple[RatingSequence, RatingSequence]:
        """Find longest positive and negative sequences."""
        max_pos = RatingSequence("positive", 0, None, None)
        max_neg = RatingSequence("negative", 0, None, None)

        for rating, group in self._group_sequential_ratings():
            group_list = list(group)
            if rating == "positive":
                max_pos = self._update_max_sequence(max_pos, group_list)
            elif rating == "negative":
                max_neg = self._update_max_sequence(max_neg, group_list)

        return max_pos, max_neg

    def calculate_average_length(self) -> float:
        """Calculate average length of sequential ratings."""
        sequences = [(rating, list(group))
                     for rating, group in self._group_sequential_ratings()
                     if rating in ["positive", "negative"]]

        total_length = sum(len(group) for _, group in sequences)
        return total_length / len(sequences) if sequences else 0

    def _group_sequential_ratings(self) -> Iterator:
        """Group ratings into sequential chunks."""
        return itertools.groupby(self.df.itertuples(), lambda x: x.rating)

    @staticmethod
    def _update_max_sequence(current: RatingSequence, group: List) -> RatingSequence:
        """Update maximum sequence if current group is longer."""
        if len(group) > current.length:
            return RatingSequence(
                rating=current.rating,
                length=len(group),
                start_id=group[0].id,
                end_id=group[-1].id
            )
        return current

    def save_results(self, max_sequences: Tuple[RatingSequence, RatingSequence],
                     avg_length: float) -> None:
        """Save results to CSV files."""
        max_pos, max_neg = max_sequences

        pd.DataFrame({
            "rating": ["positive", "negative"],
            "length": [max_pos.length, max_neg.length],
            "start_id": [max_pos.start_id, max_neg.start_id],
            "end_id": [max_pos.end_id, max_neg.end_id]
        }).to_csv(os.path.join(self.export_dir, "max_ratings.csv"), index=False)

        pd.DataFrame({"average_length": [avg_length]}).to_csv(
            os.path.join(self.export_dir, "average_length.csv"), index=False)


if __name__ == "__main__":
    analyzer = RatingsAnalyzer()
    analyzer.process()
