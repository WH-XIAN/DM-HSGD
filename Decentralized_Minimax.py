import argparse
import os
from sklearn.datasets import load_svmlight_file
import numpy as np
from mpi4py import MPI

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Decentralized Minimax Problem')
parser.add_argument('--data_path', default='./Data', type=str, help='path of data')
parser.add_argument('--data_file', default='a9a.txt', type=str, help='file of data')
parser.add_argument('--lambda2', default=0.001, type=float, help='coefficient of regularization')
parser.add_argument('--alpha', default=10.0, type=float, help='coefficient in regularization')
parser.add_argument('--lr_x', default=0.01, type=float, help='learning rate for x')
parser.add_argument('--lr_y', default=0.1, type=float, help='learning rate for y')
parser.add_argument('--beta_x', default=0.01, type=float, help='weight of SGD for x')
parser.add_argument('--beta_y', default=0.1, type=float, help='weight of SGD for y')
parser.add_argument('--bx', default=10, type=int, help='mini batch size for x')
parser.add_argument('--by', default=10, type=int, help='mini batch size for y')
parser.add_argument('--b0', default=2000, type=int, help='large batch size for the certain iteration')
parser.add_argument('--num_epochs', default=500, type=int, help='number of epochs to train')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='./result_DMHSGD.csv', type=str, help='path of output file')
# --------------------------------------------------------------------------- #


def load_data(path, file):
    source = path + '/' + file
    data = load_svmlight_file(source)
    x_raw = data[0]
    y = np.array(data[1])
    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw.todense()
    return x, y


def cal_gradient_x(data, label, x, y, lambda2, alpha):
    term1 = np.exp(- np.sum(data * x, axis=1) * label)
    term2 = - y * label * term1 / (1 + term1)
    grad1 = np.matmul(term2, data)
    denominator = (1 + alpha * x * x) * (1 + alpha * x * x)
    numerator = 2 * lambda2 * alpha * x
    grad2 = numerator / denominator
    return grad1 + grad2


def cal_gradient_y(data, label, x, y, idx, n):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label))
    grad = np.ones(n) * 1.0 / n
    grad[idx] += logistic_loss
    return grad - y


def consensus(variable, comm, rank, size):
    left = (rank + size - 1) % size
    right = (rank + 1) % size
    send_buffer = np.copy(variable)
    recv_left = np.zeros_like(send_buffer, dtype=float)
    recv_right = np.zeros_like(send_buffer, dtype=float)
    req_left = comm.Isend(send_buffer, dest=left, tag=1)
    req_right = comm.Isend(send_buffer, dest=right, tag=2)
    comm.Recv(recv_left, source=left, tag=2)
    comm.Recv(recv_right, source=right, tag=1)
    req_left.wait()
    req_right.wait()
    return (recv_left + recv_right + variable) / 3


def projection(y, n):
    y_sort = np.sort(y)
    y_sort = y_sort[::-1]
    sum_y = 0
    t = 0
    for i in range(n):
        sum_y = sum_y + y_sort[i]
        t = (sum_y - 1.0) / (i + 1)
        if i < n - 1 and y_sort[i + 1] <= t < y_sort[i]:
            break
    return np.maximum(y - t, 0)


def cal_phi(data, label, x, n, lambda2, alpha):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label))
    un = np.ones(n) * 1.0 / n
    y_star = projection(logistic_loss + un, n)
    phi = np.inner(y_star, logistic_loss) - 0.5 * np.sum((y_star - un) * (y_star - un)) + lambda2 * \
          np.sum(alpha * x * x / (alpha * x * x + 1))
    return phi


def main():
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    path = args.data_path
    datafile = args.data_file
    data, label = load_data(path, datafile)
    n, d = data.shape
    x = np.ones(d) * 1
    y = np.ones(n) * 1.0 / n

    x_old = np.copy(x)
    gx_old = np.zeros_like(x)
    vx = np.zeros_like(x)
    x_bar = np.copy(x)
    y_old = np.copy(y)
    gy_old = np.zeros_like(y)
    vy = np.zeros_like(y)

    alpha = args.alpha
    lambda2 = args.lambda2

    if rank == 0:
        phi = cal_phi(data, label, x, n, lambda2, alpha)
        if not os.path.exists(args.out_fname):
            with open(args.out_fname, 'w') as f:
                print('{ep:d},{t:.3f},{phi:.8f}'.format(ep=0, t=0, phi=phi), file=f)

    elapsed_time = 0.0
    for epoch in range(args.num_epochs):
        t_begin = MPI.Wtime()
        # decentralized minimax storm
        if epoch == 0:
            id0 = np.random.randint(0, n, args.b0)
            # id0 = range(n)
            gx = cal_gradient_x(data[id0, :], label[id0], x, y[id0], lambda2, alpha)
            gy = cal_gradient_y(data[id0, :], label[id0], x, y, id0, n)
        else:
            idx = np.random.randint(0, n, args.bx)
            idy = np.random.randint(0, n, args.by)
            gx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], lambda2, alpha) + (1 - args.beta_x) * \
                 (gx_old - cal_gradient_x(data[idx, :], label[idx], x_old, y_old[idx], lambda2, alpha))
            gy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n) + (1 - args.beta_y) * \
                 (gy_old - cal_gradient_y(data[idy, :], label[idy], x_old, y_old, idy, n))
        # gradient tracking
        vx = vx + gx - gx_old
        vy = vy + gy - gy_old
        vx = consensus(vx, comm, rank, size)
        vy = consensus(vy, comm, rank, size)
        gx_old = np.copy(gx)
        gy_old = np.copy(gy)
        # update x and y
        x_old = np.copy(x)
        x = x - args.lr_x * vx
        x = consensus(x, comm, rank, size)
        y_old = np.copy(y)
        y = y + args.lr_y * vy
        y = consensus(y, comm, rank, size)
        y = projection(y, n)

        t_end = MPI.Wtime()
        elapsed_time += (t_end - t_begin)
        if (epoch + 1) % args.print_freq == 0:
            # compute x_bar, y_bar on rank 0
            comm.Reduce(x, x_bar, op=MPI.SUM)
            if rank == 0:
                x_bar = x_bar / size
                phi = cal_phi(data, label, x_bar, n, lambda2, alpha)
                with open(args.out_fname, '+a') as f:
                    print('{ep:d},{t:.3f},{phi:.8f}'.format(ep=epoch + 1, t=elapsed_time, phi=float(phi)), file=f)
        comm.barrier()


if __name__ == '__main__':
    main()
