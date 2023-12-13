import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import random
import numpy as np
import torch.nn.functional as F


def DICE_attack(graph, target_node, delta):
    """
    DICE attack: randomly disconnects d edges from the test node v to same-class nodes
    and connects b edges from v to different-class nodes.
    """
    d = 0
    b = delta
    neighbors = graph.successors(target_node)
    # same_class_nodes = [neighbor for neighbor in neighbors if graph.ndata['label'][neighbor] == graph.ndata['label'][target_node]]
    different_class_nodes = [node for node in graph.nodes() if node not in neighbors and graph.ndata['label'][node] != graph.ndata['label'][target_node]]
    
    # # Disconnect d edges to same-class nodes
    # edges_to_disconnect = np.random.choice(same_class_nodes, size=min(d, len(same_class_nodes)), replace=False)
    # graph.remove_edges(target_node, edges_to_disconnect)
    
    # Connect b edges to different-class nodes
    edges_to_connect = np.random.choice(different_class_nodes, size=min(b, len(different_class_nodes)), replace=False)
    graph.add_edges(target_node, edges_to_connect)

def l2_weak_attack(graph, target_node):
    """
    l2-weak attack: connects a node to its most-similar different class nodes in feature space.
    """
    features = graph.ndata['feat']
    target_features = features[target_node]
    distances = F.pairwise_distance(features, target_features, p=2)
    
    # Sort nodes by distance
    sorted_nodes = torch.argsort(distances)
    
    # Connect to most-similar different-class nodes
    for node in sorted_nodes:
        if graph.ndata['label'][node] != graph.ndata['label'][target_node]:
            graph.add_edge(target_node, node)
            break

def l2_strong_attack(graph, target_node):
    """
    l2-strong attack: connects a target node to its most-dissimilar different class nodes in feature space.
    """
    features = graph.ndata['feat']
    target_features = features[target_node]
    distances = F.pairwise_distance(features, target_features, p=2)
    
    # Sort nodes by distance in descending order
    sorted_nodes = torch.argsort(distances, descending=True)
    
    # Connect to most-dissimilar different-class nodes
    for node in sorted_nodes:
        if graph.ndata['label'][node] != graph.ndata['label'][target_node]:
            graph.add_edge(target_node, node)
            break



def get_dataset(dataset, pe_dim, attack, attperc, TrainTest):
    assert attperc is int

    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:



        file_path = "dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]


        if attack == "none":
            pass
        elif attack == "dice":
            if TrainTest == 'train':
                nodes = np.random.choice(idx_train, int((attperc/100)*len(idx_train)))
                for node in nodes:
                    DICE_attack(graph, node, delta=2)
            elif TrainTest == 'test':
                nodes = np.random.choice(idx_test, int((attperc/100)*len(idx_test)))
                for node in nodes:
                    DICE_attack(graph, node, delta=2)
        elif attack == "l2_weak":
            if TrainTest == 'train':
                nodes = np.random.choice(idx_train, int((attperc/100)*len(idx_train)))
                for node in nodes:
                    l2_weak_attack(graph, node)
            elif TrainTest == 'test':
                nodes = np.random.choice(idx_test, int((attperc/100)*len(idx_test)))
                for node in nodes:
                    l2_weak_attack(graph, node)
        elif attack == "l2_strong":
            if TrainTest == 'train':
                nodes = np.random.choice(idx_train, int((attperc/100)*len(idx_train)))
                for node in nodes:
                    l2_strong_attack(graph, node)
            elif TrainTest == 'test':
                nodes = np.random.choice(idx_test, int((attperc/100)*len(idx_test)))
                for node in nodes:
                    l2_strong_attack(graph, node)

        
        graph = dgl.to_bidirected(graph)
        adj = graph.adj() #data_list[0]
        features = graph.ndata['feat'] #data_list[1]
        labels = graph.ndata['label']  #data_list[2]

        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
     
        features = torch.cat((features, lpe), dim=1)


    return adj, features, labels, idx_train, idx_val, idx_test




