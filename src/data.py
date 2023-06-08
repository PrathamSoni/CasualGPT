import pandas as pd

TOPOSORT = "chemical, gene, haplotype, variant disease"

valid_edges = [('Gene', 'Disease'), ('Haplotype', 'Disease'), ('Gene', 'Haplotype'), ('Gene', 'Variant'),
               ('Chemical', 'Variant'), ('Haplotype', 'Variant'), ('Chemical', 'Haplotype'), ('Haplotype', 'Haplotype'),
               ('Gene', 'Gene'), ('Chemical', 'Gene'), ('Variant', 'Disease'), ('Chemical', 'Chemical'),
               ('Disease', 'Disease'), ('Variant', 'Variant')]


def read_data():
    return pd.read_csv('../data/relationships.tsv', sep="\t")


def process_data():
    data = read_data()
    data = data[data.Association == "associated"]  # 74280
    data = data[data[["Entity1_type", "Entity2_type"]].apply(tuple, axis=1).isin(valid_edges)]  # 38486
    data = data[["Entity1_name", "Entity1_type", "Entity2_name", "Entity2_type"]]
    return data


def check_dag():
    data = process_data()
