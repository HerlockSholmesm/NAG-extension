import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
# from torch_sparse import spspmm
import os
import re
import copy
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import dgl
from sklearn.metrics import confusion_matrix, f1_score



def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def confusion_matrix_metric(output_before_attack, output_after_attack, labels):
    preds_before_attack = output_before_attack.max(1)[1].type_as(labels)
    preds_after_attack = output_after_attack.max(1)[1].type_as(labels)

    cm = confusion_matrix(labels.cpu().numpy(), preds_before_attack.cpu().numpy())
    cm_after_attack = confusion_matrix(labels.cpu().numpy(), preds_after_attack.cpu().numpy())

    return cm, cm_after_attack

def robustness_index(output_before_attack, output_after_attack, labels):
    acc_before_attack = accuracy(output_before_attack, labels)
    acc_after_attack = accuracy(output_after_attack, labels)

    ri = (acc_before_attack - acc_after_attack) / acc_before_attack
    return ri.item()

def node_wise_robustness(output_before_attack, output_after_attack, labels):
    preds_before_attack_probs = output_before_attack.softmax(dim=1)
    preds_after_attack_probs = output_after_attack.softmax(dim=1)

    correct_class_probs_before_attack = preds_before_attack_probs[range(len(labels)), labels]
    correct_class_probs_after_attack = preds_after_attack_probs[range(len(labels)), labels]

    nwr = (correct_class_probs_before_attack - correct_class_probs_after_attack) / correct_class_probs_before_attack
    return nwr.cpu().numpy()

def f1_score_metric(output, labels):
    preds = output.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()

    f1 = f1_score(labels, preds, average='weighted')
    return f1

# Batch versions

def confusion_matrix_metric_batch(output_before_attack, output_after_attack, labels):
    preds_before_attack = output_before_attack.max(1)[1].type_as(labels)
    preds_after_attack = output_after_attack.max(1)[1].type_as(labels)

    cm = confusion_matrix(labels.cpu().numpy().ravel(), preds_before_attack.cpu().numpy().ravel())
    cm_after_attack = confusion_matrix(labels.cpu().numpy().ravel(), preds_after_attack.cpu().numpy().ravel())

    return cm, cm_after_attack

def robustness_index_batch(output_before_attack, output_after_attack, labels):
    acc_before_attack = accuracy_batch(output_before_attack, labels)
    acc_after_attack = accuracy_batch(output_after_attack, labels)

    ri = (acc_before_attack - acc_after_attack) / acc_before_attack
    return ri.item()

def node_wise_robustness_batch(output_before_attack, output_after_attack, labels):
    preds_before_attack_probs = output_before_attack.softmax(dim=1)
    preds_after_attack_probs = output_after_attack.softmax(dim=1)

    correct_class_probs_before_attack = preds_before_attack_probs[range(len(labels)), labels]
    correct_class_probs_after_attack = preds_after_attack_probs[range(len(labels)), labels]

    nwr = (correct_class_probs_before_attack - correct_class_probs_after_attack) / correct_class_probs_before_attack
    return nwr.cpu().numpy()

def f1_score_metric_batch(output, labels):
    preds = output.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()

    f1 = f1_score(labels, preds, average='weighted')
    return f1



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[1]))

    return sp_matrix



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc



def re_features(adj, features, K):
    #feature matrix after propagation ,size= (N, 1, K+1, d)
    nodes_features = torch.empty(features.shape[0], 1, K+1, features.shape[1])

    for i in range(features.shape[0]):

        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):

            nodes_features[index, 0, i + 1, :] = x[index]        

    nodes_features = nodes_features.squeeze()


    return nodes_features


def nor_matrix(adj, a_matrix):

    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix





