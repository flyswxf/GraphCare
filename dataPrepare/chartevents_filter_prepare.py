import argparse
import csv
import os
import pickle
from typing import Callable, Iterable, List, Set

from pyhealth.data import Patient, Visit
from pyhealth.datasets import MIMIC3Dataset


def load_itemids(path: str) -> Set[str]:
    """Loads item IDs from a txt or csv file.

    Supports:
    - Plain text: one itemid per line
    - CSV: expects a column named 'itemid' (case-insensitive)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ItemID file does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    itemids: Set[str] = set()

    if ext in (".csv", ".tsv"):  # CSV with a column named 'itemid'
        with open(path, newline="", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            sample = f.read(2048)
            f.seek(0)
            dialect = sniffer.sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
            # find the itemid column in a case-insensitive way
            cols = {c.lower(): c for c in reader.fieldnames or []}
            if "itemid" not in cols:
                raise ValueError("CSV file must contain a column named 'itemid'")
            colname = cols["itemid"]
            for row in reader:
                v = str(row[colname]).strip()
                if v:
                    itemids.add(v)
    else:
        # plain text: one itemid per line (assuming first word is the itemid)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:  # 确保行不为空
                    v = parts[0].strip()  # 只取第一个词作为itemid
                    if v:
                        itemids.add(v)

    return itemids


def make_chartevents_filter_task(valid_itemids: Set[str]) -> Callable[[Patient], List[dict]]:
    """Factory to create a task function that filters CHARTEVENTS by given itemids.

    Only samples whose CHARTEVENTS contain at least one valid itemid will be added.
    The returned sample dict includes filtered 'chartevents' along with conditions/procedures/drugs.
    """

    def task_fn(patient: Patient) -> List[dict]:
        samples: List[dict] = []
        for i in range(len(patient)):
            visit: Visit = patient[i]
            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs = visit.get_code_list(table="PRESCRIPTIONS")
            chartevents = visit.get_code_list(table="CHARTEVENTS")

            # exclude: visits without condition, procedure, or drug code (keep parity with existing tasks)
            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            # filter chartevents by whitelist
            # codes can be numeric or string; normalize to str for comparison
            filtered_chartevents = [str(c) for c in chartevents if str(c) in valid_itemids]

            # require at least one valid chartevent before adding the sample
            if len(filtered_chartevents) == 0:
                continue

            samples.append(
                {
                    "visit_id": visit.visit_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures,
                    "drugs": drugs,
                    "chartevents": filtered_chartevents,
                }
            )

        return samples if len(samples) > 0 else []

    return task_fn


def build_dataset(mimic3_root: str, itemid_file: str):
    # 1) load itemids
    valid_itemids = load_itemids(itemid_file)
    if not valid_itemids:
        raise ValueError("No valid itemids loaded from file; please check the file content.")

    # 2) init dataset with CHARTEVENTS table included
    ds = MIMIC3Dataset(
        root=mimic3_root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS", "CHARTEVENTS"],
    )

    # 3) set custom task
    sample_dataset = ds.set_task(make_chartevents_filter_task(valid_itemids))
    return sample_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-III dataset with CHARTEVENTS filtering by itemids.")
    parser.add_argument(
        "--root",
        type=str,
        default="./data/mimic3/",
        help="Path to the MIMIC-III root directory.",
    )
    parser.add_argument(
        "--itemid_file",
        type=str,
        default="./dataPrepare/match_stats/itemids.csv",
        help="Path to the file containing desired itemids (txt: one per line; csv: column 'itemid').",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/addCHARTEVENT/sample_dataset.pkl",
        help="Optional path to save the prepared sample_dataset as a pickle file.",
    )

    args = parser.parse_args()

    sample_dataset = build_dataset(args.root, args.itemid_file)

    print(f"Prepared sample dataset with {len(sample_dataset)} patient samples after filtering.")

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "wb") as f:
            pickle.dump(sample_dataset, f)
        print(f"Saved sample_dataset to: {args.save_path}")


if __name__ == "__main__":
    main()