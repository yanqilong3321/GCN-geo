import numpy as np
import scipy.sparse as sp
import torch
import _pickle as cPickle
import gzip
import logging
from haversine import haversine


def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    print("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: %.2f"% acc_at_161)

    metirc = {"Mean":int(np.mean(distances)) ,
               "Median" : int(np.median(distances)) ,
              "Acc@161":int(acc_at_161)}

    return metirc
    # print("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(
    #         int(acc_at_161)))
    # return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_coradata(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),         # edges_unordered->edges likes:[paper_id,neigbo_paper_id]->[id,neigbo_paper_id]
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # print(type(adj),type(features))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # features = torch.FloatTensor(np.array(features.todense()))
    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_geodata():

    # #完整数据
    data= load_obj('/home/yanqilong/workspace/geographconv-master_edit/data/cmu/dump.pkl')
    # our graph
    # data = load_obj('/home/yanqilong/workspace/GCN-geo/data/cmu-dump-geograph-Tfidf.pkl')
    # data = load_obj('/sdc/yanqilong/workspace/Home-Computer/geographconv-master_edit/data/na/dump.pkl')

    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation,_ = data

    idx_train = range(0, X_train.shape[0])
    idx_val = range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])
    idx_test = range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])

    # X = sp.vstack([X_train, X_dev, X_test])
    # X_train = X_train.astype('float16')
    # X_dev = X_dev.astype('float16')
    # X_test = X_test.astype('float16')
    X_train = X_train.todense().A
    X_dev = X_dev.todense().A
    X_test = X_test.todense().A
    A  = A.todense().A
    X = np.vstack([X_train, X_dev, X_test])
    Y = np.hstack((Y_train, Y_dev, Y_test))
    Y = Y.astype('int32')
    X = X.astype('float32')
    A = A.astype('float32')

    adj  = torch.FloatTensor(A)
    # adj = sparse_mx_to_torch_sparse_tensor(A)
    # features = sparse_mx_to_torch_sparse_tensor(X)
    # train_features = sparse_mx_to_torch_sparse_tensor(X_train)
    features = torch.FloatTensor(X)
    train_features = torch.FloatTensor(X_train)
    labels = torch.LongTensor(Y)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(X.shape)
    return  adj, features, labels, idx_train, idx_val, idx_test,U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation,train_features
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_obj(filename, serializer=cPickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin,encoding='iso-8859-1')  #,encoding='iso-8859-1'
    return obj
