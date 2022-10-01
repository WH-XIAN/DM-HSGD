import argparse
import os
from sklearn.datasets import load_svmlight_file
import numpy as np
import time
import random

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Perturbed Minimax Algorithm')
parser.add_argument('--data_folder', default='Data', type=str, help='path of data folder')
parser.add_argument('--data_file', default='a9a.txt', type=str, help='name of data file')
parser.add_argument('--lambda2', default=0.001, type=float, help='coefficient of regularization')
parser.add_argument('--alpha', default=10.0, type=float, help='coefficient in regularization')
parser.add_argument('--algorithm', default=4, type=int, help='optimization algorithm')
parser.add_argument('--lr_x', default=0.001, type=float, help='learning rate for x in descent phase')
parser.add_argument('--lr_x_h', default=0.1, type=float, help='learning rate for x in escaping phase')
parser.add_argument('--lr_y', default=0.01, type=float, help='learning rate for y')
parser.add_argument('--beta_x', default=0.01, type=float, help='weight of SGD for x')
parser.add_argument('--beta_y', default=0.1, type=float, help='weight of SGD for y')
parser.add_argument('--bx', default=40, type=int, help='mini batch size for x')
parser.add_argument('--by', default=40, type=int, help='mini batch size for y')
parser.add_argument('--num_epochs', default=500, type=int, help='number of epochs to train')
parser.add_argument('--q', default=25, type=int, help='nested loops for variance reduction')
parser.add_argument('--K', default=5, type=int, help='nested loops for maximizer')
parser.add_argument('--init_y', default=200, type=int, help='steps to initialize y in SREDA')
parser.add_argument('--epsilon_sreda', default=0.05, type=float, help='parameter to control stepsize in SREDA')
parser.add_argument('--epsilon_esc', default=0.0005, type=float, help='threshold to switch escaping phase')
parser.add_argument('--esc_max', default=20, type=int, help='maximum steps of escaping phase')
parser.add_argument('--esc_dist', default=0.01, type=float, help='average moving distance of escaping phase')
parser.add_argument('--radius', default=0.01, type=float, help='perturbation radius')
parser.add_argument('--print_freq', default=10, type=int, help='frequency to print train stats')
parser.add_argument('--out_fname', default='PMHSGD.csv', type=str, help='path of output file')
# --------------------------------------------------------------------------- #


def load_data(folder, file):
    source = os.path.join(folder, file)
    '''
    if file == 'rcv1.txt':
        length = round(1.2153e9 / size)
        offset = rank * length
        data = load_svmlight_file(source, n_features=47237, offset=offset, length=length)
    '''
    data = load_svmlight_file(source)
    x_raw = data[0]
    y = np.array(data[1])
    x = np.ones([x_raw.shape[0], x_raw.shape[1] + 1])
    x[:, :-1] = x_raw.todense()
    norm = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    x = x / norm
    return x, y


def cal_gradient_x(data, label, x, y, n, lambda2, alpha):
    term1 = np.exp(- np.sum(data * x, axis=1) * label)
    term2 = - y * label * term1 / (1 + term1)
    # grad1 = np.matmul(term2, data) / data.shape[0]
    grad1 = np.matmul(term2, data) * n / len(y)
    denominator = (1 + alpha * x * x) * (1 + alpha * x * x)
    numerator = 2 * lambda2 * alpha * x
    grad2 = numerator / denominator
    return grad1 + grad2


def cal_gradient_y(data, label, x, y, idx, n):
    logistic_loss = np.log(1 + np.exp(- np.sum(data * x, axis=1) * label)) * n / len(idx)
    grad = np.ones(n) * 1.0 / n
    grad[idx] += logistic_loss
    return grad - y


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
    grad = cal_gradient_x(data, label, x, y_star, n, lambda2, alpha)
    grad_norm = np.sqrt(np.sum(grad * grad)) * 10000 / n
    return phi, grad_norm


def main():
    args = parser.parse_args()
    folder = args.data_folder
    datafile = args.data_file
    data, label = load_data(folder, datafile)
    n, d = data.shape
    # n = 5000
    # data = data[range(n), :]
    # label = label[range(n)]
    x = np.ones(d) * 0.1
    y = np.ones(n) * 1.0 / n

    x_old = np.copy(x)
    gx_old = np.zeros_like(x)
    vx = np.zeros_like(x)
    y_old = np.copy(y)
    gy_old = np.zeros_like(y)
    vy = np.zeros_like(y)

    alpha = args.alpha
    lambda2 = args.lambda2

    print('eta_x: %f, eta_y: %f, eta_h: %f' % (args.lr_x, args.lr_y, args.lr_x_h))
    print('radius: %f, D_bar: %f, T_thresh: %d, epsilon: %f' % (args.radius, args.esc_dist, args.esc_max, args.epsilon_esc))

    phi, grad_norm = cal_phi(data, label, x, n, lambda2, alpha)
    if not os.path.exists(args.out_fname):
        with open(args.out_fname, 'w') as f:
            print('Epoch,Time,IFO,Phi,Grad_Norm', file=f)
            print('{ep:d},{t:.3f},{ifo:.3f},{phi:.5f},{grad:.5f}'.format(ep=0, t=0, ifo=0, phi=phi, grad=grad_norm), file=f)

    elapsed_time = 0.0
    oracle = 0
    dk = 1
    escape = False
    esc = 0
    for epoch in range(args.num_epochs):
        t_begin = time.time()
        if args.algorithm == 1:
            # minimax storm
            if epoch == 0:
                id0 = range(n)
                gx = cal_gradient_x(data[id0, :], label[id0], x, y[id0], n, lambda2, alpha)
                gy = cal_gradient_y(data[id0, :], label[id0], x, y, id0, n)
                oracle += 2 * n
            else:
                idx = np.random.randint(0, n, args.bx)
                idy = np.random.randint(0, n, args.by)
                gx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha) + (1 - args.beta_x) * \
                     (gx_old - cal_gradient_x(data[idx, :], label[idx], x_old, y_old[idx], n, lambda2, alpha))
                gy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n) + (1 - args.beta_y) * \
                     (gy_old - cal_gradient_y(data[idy, :], label[idy], x_old, y_old, idy, n))
                oracle = oracle + 2 * args.bx + 2 * args.by
            # gradient tracking
            vx = vx + gx - gx_old
            vy = vy + gy - gy_old
            gx_old = np.copy(gx)
            gy_old = np.copy(gy)
            # update x and y
            x_old = np.copy(x)
            x = x - args.lr_x * vx
            y_old = np.copy(y)
            y = y + args.lr_y * vy
            y = projection(y, n)
        elif args.algorithm == 2:
            # SGDA
            idx = np.random.randint(0, n, args.bx)
            idy = np.random.randint(0, n, args.by)
            gx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha)
            gy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n)
            oracle = oracle + args.bx + args.by
            # update x and y
            x = x - args.lr_x * gx
            y = y + args.lr_y * gy
            y = projection(y, n)
        elif args.algorithm == 3:
            # GDA
            if epoch >= 4000:
                # dk = 0.8
                dk = 1
            idx = range(n)
            idy = range(n)
            gx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha)
            gy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n)
            oracle += 2 * n
            # update x and y
            x = x - args.lr_x * dk * gx
            y = y + args.lr_y * gy
            y = projection(y, n)
        else:
            # SREDA
            if epoch == 0:
                # initialization
                for k in range(args.init_y):
                    if k % args.q == 0:
                        idy = range(n)
                        vy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n)
                        oracle += n
                    else:
                        idy = np.random.randint(0, n, args.by)
                        vy = vy + cal_gradient_y(data[idy, :], label[idy], x, y, idy, n) - \
                             cal_gradient_y(data[idy, :], label[idy], x, y_old, idy, n)
                        oracle += 2 * args.by
                    y_old = np.copy(y)
                    y = y + args.lr_y * vy
                    y = projection(y, n)
            if epoch % args.q == 0:
                idx = range(n)
                idy = range(n)
                vx = cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha)
                vy = cal_gradient_y(data[idy, :], label[idy], x, y, idy, n)
                oracle += 2 * n
            else:
                idx = np.random.randint(0, n, args.bx)
                idy = np.random.randint(0, n, args.by)
                vx = vx + cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha) - \
                     cal_gradient_x(data[idx, :], label[idx], x_old, y[idx], n, lambda2, alpha)
                vy = vy + cal_gradient_y(data[idy, :], label[idy], x, y, idy, n) - \
                     cal_gradient_y(data[idy, :], label[idy], x_old, y, idy, n)
                oracle = oracle + 2 * args.bx + 2 * args.by
                y_old = np.copy(y)
                y = y + args.lr_y * vy
                y = projection(y, n)
                # update y and estimate y^*(x)
                for k in range(args.K):
                    idx = np.random.randint(0, n, args.bx)
                    idy = np.random.randint(0, n, args.by)
                    vx = vx + cal_gradient_x(data[idx, :], label[idx], x, y[idx], n, lambda2, alpha) - \
                         cal_gradient_x(data[idx, :], label[idx], x, y_old[idx], n, lambda2, alpha)
                    vy = vy + cal_gradient_y(data[idy, :], label[idy], x, y, idy, n) - \
                         cal_gradient_y(data[idy, :], label[idy], x, y_old, idy, n)
                    oracle = oracle + 2 * args.bx + 2 * args.by
                    y_old = np.copy(y)
                    y = y + args.lr_y * vy
                    y = projection(y, n)

            # update x
            vx_norm = np.sqrt(np.sum(vx * vx))
            if not escape:
                if vx_norm > args.epsilon_esc:
                    x_old = np.copy(x)
                    x = x - np.min([1, args.epsilon_sreda / vx_norm]) * args.lr_x * vx
                    # x = x - (args.lr_x / vx_norm) * vx
                else:
                    escape = True
                    esc = 0
                    dist = 0
                    # draw a perturbation from a small ball
                    direction = np.random.rand(d)
                    direction = direction / np.sqrt(np.sum(direction * direction))
                    r_coeff = random.random()
                    x_old = np.copy(x)
                    x = x + r_coeff * args.radius * direction
            else:
                esc += 1
                aug = args.lr_x_h * args.lr_x_h * vx_norm * vx_norm
                if dist + aug > esc * args.esc_dist:
                    lr_pull = np.sqrt((args.esc_dist - dist) / (vx_norm * vx_norm))
                    x_old = np.copy(x)
                    x = x - lr_pull * vx
                    escape = False
                else:
                    dist = dist + aug
                    x_old = np.copy(x)
                    x = x - args.lr_x_h * vx
                    if esc >= args.esc_max:
                        escape = False

        t_end = time.time()
        elapsed_time += (t_end - t_begin)
        if (epoch + 1) % args.print_freq == 0:
            phi, grad_norm = cal_phi(data, label, x, n, lambda2, alpha)
            with open(args.out_fname, '+a') as f:
                print('{ep:d},{t:.3f},{ifo:.3f},{phi:.5f},{grad:.5f}'
                      .format(ep=epoch + 1, t=elapsed_time, ifo=oracle/n, phi=float(phi), grad=grad_norm), file=f)


if __name__ == '__main__':
    main()
