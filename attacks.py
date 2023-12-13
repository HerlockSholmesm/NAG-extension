import torch
import torch.nn.functional as F
import dgl
import numpy as np

def DICE_attack(graph, target_node, delta):
    """
    DICE attack: randomly disconnects d edges from the test node v to same-class nodes
    and connects b edges from v to different-class nodes.
    """
    d = 0
    b = delta
    neighbors = graph.successors(target_node)
    same_class_nodes = [neighbor for neighbor in neighbors if graph.ndata['label'][neighbor] == graph.ndata['label'][target_node]]
    different_class_nodes = [node for node in graph.nodes() if node not in neighbors and graph.ndata['label'][node] != graph.ndata['label'][target_node]]
    
    # Disconnect d edges to same-class nodes
    edges_to_disconnect = np.random.choice(same_class_nodes, size=min(d, len(same_class_nodes)), replace=False)
    graph.remove_edges(target_node, edges_to_disconnect)
    
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

