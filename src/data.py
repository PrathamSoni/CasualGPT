import random
import torch

import pandas as pd
import networkx as nx

random.seed(1234)
torch.manual_seed(1234)

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
    data = data[data["Entity1_name"] < data["Entity2_name"]]  # 13343
    return data


def build_graph():
    df = process_data()
    G = nx.DiGraph()
    for entity1, _, entity2, _ in df.values:
        G.add_edge(entity1, entity2)
    return G


def splits():
    G = build_graph()
    num_nodes = G.number_of_nodes()  # 6695, 13343
    nodes = G.nodes()
    train_nodes = random.sample(nodes, int(0.8 * num_nodes))
    test_nodes = nodes - set(train_nodes)
    G_train = nx.induced_subgraph(G, train_nodes).copy()
    G_test = nx.induced_subgraph(G, test_nodes).copy()
    return G_train, G_test


def get_edges(G):
    positive_edges = list(G.edges())
    num_edges = len(positive_edges)
    negative_edges = list(nx.non_edges(G))
    negative_edges = random.sample(negative_edges, min(num_edges, len(negative_edges)))
    return positive_edges, negative_edges


def get_datasets():
    G_train, G_test = splits()
    return get_edges(G_train), get_edges(G_test)


def check_dag():
    G = build_graph()
    assert len(list(nx.simple_cycles(G))) == 0


if __name__ == "__main__":
    check_dag()
