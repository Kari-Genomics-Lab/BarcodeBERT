#!/usr/bin/env python

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse


def split_df(filename):

    df_dtypes = {
        "processid": "str",
        "sampleid": "str",
        "chunk": "uint8",
        "phylum": "category",
        "class": "category",
        "order": "category",
        "family": "category",
        "subfamily": "category",
        "genus": "category",
        "species": "category",
        "dna_bin": "category",
        "dna_barcode": str,
        "split": "category",
        "country": "category",
        "province/state": "category",
        "coord-lat": float,
        "coord-lon": float,
        "surface_area": float,
        "bioscan1M_index": "Int64",
        "label_was_inferred": "uint8",
    }

    label_cols = ["phylum", "class", "order", "family", "subfamily", "genus", "species", "dna_bin"]
    df_usecols = ["chunk", "dna_barcode", "split"] + label_cols

    bioscan_5M = pd.read_csv(filename, dtype=df_dtypes, usecols=df_usecols)

    # Convert categorical columns to int codes which we can use as training targets
    for c in label_cols:
        bioscan_5M[c + "_index"] = bioscan_5M[c].cat.codes

    bioscan_5M = bioscan_5M.rename(
        columns={
            "order": "order_name",
            "family": "family_name",
            "subfamily": "subfamily_name",
            "genus": "genus_name",
            "species": "species_name",
            "dna_barcode": "nucleotides",
        }
    )
    bioscan_5M["nucleotides"] = bioscan_5M["nucleotides"].str.rstrip("N")

    # Check how many samples  with and w/o DNA sequences
    n_seq = sum(bioscan_5M["nucleotides"].notna() & (bioscan_5M["nucleotides"] != "no_data"))
    n_no_seq = sum(bioscan_5M["nucleotides"].isna() | (bioscan_5M["nucleotides"] == "no_data"))

    print(f"{n_seq} out of {len(bioscan_5M)} sequences contain a DNA identifier ({100*n_seq/len(bioscan_5M):.2f}%)")
    print(
        f"{n_no_seq} out of {len(bioscan_5M)} sequences do not contain a DNA identifier ({100*n_no_seq/len(bioscan_5M):.2f}%)"
    )

    # Finding duplicated sequences
    duplicates = bioscan_5M[
        bioscan_5M["nucleotides"].notna()
        & (bioscan_5M.nucleotides != "no_data")
        & (bioscan_5M.duplicated(subset=["nucleotides"]))
    ]

    if len(duplicates) > 0:
        print(f"There are {len(duplicates)} specimens with a repeated  DNA sequence")
        print()

        # Duplicated sequences with different sequence identifiers
        duplicates = bioscan_5M[
            bioscan_5M["nucleotides"].notna()
            & (bioscan_5M["nucleotides"] != "no_data")
            & (bioscan_5M.duplicated(subset=["nucleotides"], keep=False))
        ]
        inconsistent_duplicates = duplicates.groupby("nucleotides").filter(lambda x: x["species_name"].nunique() > 1)
        sorted_duplicates = inconsistent_duplicates.sort_values(by="nucleotides")
        print(sorted_duplicates)

        # Drop duplicated sequences and get dataset ready to produce DNA splits
        bioscan_5M.drop_duplicates(subset=["nucleotides"], inplace=True)

    # Distribution of samples in each split
    print(bioscan_5M.groupby("split").count())
    print()

    # Get the individual dataframes of interest
    pretrain = bioscan_5M[bioscan_5M["split"].isin(["pretrain", "other_heldout"])]
    print(f"There are {len(pretrain)} sequences in the pretrain split")
    print()

    # Get the individual dataframes of interest
    train = bioscan_5M[bioscan_5M["split"] == "train"]
    print(f"There are {len(train)} sequences in the train split")
    print()

    # Get the individual dataframes of interest
    test = bioscan_5M[bioscan_5M["split"].isin(["test", "test_seen"])]
    print(f"There are {len(test)} sequenes in the test_seen split")
    print()

    # Get the individual dataframes of interest
    unseen = bioscan_5M[bioscan_5M["split"] == "test_unseen"]
    print(f"There are {len(unseen)} sequences in the test_unseen split")
    print()

    # Get unique species
    pretrain_species = pretrain["species_name"].unique()
    train_species = train["species_name"].unique()
    test_species = test["species_name"].unique()
    unseen_species = unseen["species_name"].unique()
    print(len(unseen_species))

    # Number of repeated species from "test" split in pretrain split
    intersection_records = set(train_species).intersection(set(test_species))
    print(f"There are { len(intersection_records)} repeated species from test split in train split")
    print()

    # Number of repeated species from "unseen" split in "train" split
    intersection_records = set(train_species).intersection(set(unseen_species))
    print(f"There are { len(intersection_records)} repeated species from unseen split in train split")
    print()

    # Number of repeated species from "unseen" split in "pretrain" split
    intersection_records = set(pretrain_species).intersection(set(unseen_species))
    print(f"There are { len(intersection_records)} repeated species from unseen split in pretrain split")
    print()

    # Get unique genera
    train_genera = train["genus_name"].unique()
    test_genera = test["genus_name"].unique()
    unseen_genera = unseen["genus_name"].unique()

    intersection_records = set(train_genera).intersection(set(test_genera))
    print(f"There are {len(intersection_records)} out {len(test_genera)} genera from test split in train split")
    print()

    intersection_records = set(train_genera).intersection(set(unseen_genera))
    print(f"There are {len(intersection_records)} out {len(unseen_genera)} genera from unseen split in train split")
    print()

    # Save individual .csv files for compatibility
    pretrain.to_csv("pre_training.csv", index=False)
    train.to_csv("supervised_train.csv", index=False)
    test.to_csv("supervised_test.csv", index=False)
    unseen.to_csv("unseen.csv", index=False)

    # Don't forget validation
    bioscan_5M[bioscan_5M["split"] == "val"].to_csv("supervised_val.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, action="store")
    args = parser.parse_args()

    split_df(args.file)


if __name__ == "__main__":
    main()
