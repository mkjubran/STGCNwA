import numpy as np
import torch

def get_graph_data(num_node):
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                      (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                      (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                      (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                      (22, 23), (23, 8), (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link

    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    A2 = np.zeros((num_node, num_node))
    for root in range(A.shape[0]):
        for neighbor in range(A.shape[0]):
            if A[root, neighbor] == 1:
                for nn in range(A.shape[0]):
                    if A[neighbor, nn] == 1:
                        A2[root, nn] = 1

    bias_mat_1 = np.where(A != 0, 0.0, -1e9).astype(np.float32)
    bias_mat_2 = np.where(A2 != 0, A2, -1e9).astype(np.float32)

    AD = torch.tensor(A, dtype=torch.float32)
    AD2 = torch.tensor(A2, dtype=torch.float32)
    bias_mat_1 = torch.nn.Parameter(torch.tensor(bias_mat_1, dtype=torch.float32), requires_grad=True)
    bias_mat_2 = torch.nn.Parameter(torch.tensor(bias_mat_2, dtype=torch.float32), requires_grad=True)

    return AD, AD2, bias_mat_1, bias_mat_2
