import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt


def loss_fn(outputs, data, w, res_q, res_dq, res_dqx, args):

    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    gradients = torch.autograd.grad(outputs=outputs, inputs=data,
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
    grad_x = gradients[:, :ndim]
    grad_v = gradients[:, ndim:]
    '''
    loss = (kbt*torch.sum(grad_v**2)+torch.sum(outputs**2))/2 \
            +torch.sum(grad_v*res_dq)+torch.sum(outputs*res_q)
    '''
    # print(grad_v.shape,res_dq.shape)
    # print(outputs.shape,res_q.shape)
    if res_q.shape != (res_q.shape[0], 1):
        res_q = res_q.unsqueeze(1)
    if w.shape != (w.shape[0], 1):
        w = w.unsqueeze(1)

    # print(grad_v.shape,res_dq.shape,w.shape)
    # print(outputs.shape,res_q.shape,w.shape)
    loss = (omega * kbt * torch.sum(w * grad_v**2) + lam * torch.sum(w * outputs**2) + eta * torch.sum(w * grad_x**2)) / 2 \
        + torch.sum(w * grad_v * res_dq) + torch.sum(w * outputs *
                                                     res_q) + torch.sum(w * grad_x * res_dqx)

    # loss = loss / data.shape[0]
    # print(loss)

    return loss


b_lossfn = torch.nn.MSELoss()


def weight_update(model, data, w, dU, batchsize, args, device, threshold=1,):
    dataset = [data, dU]
    batches = split(dataset, batchsize, shuffle=False)
    pinn_l_s = []
    for d, du in batches:
        d = d.to(device)
        d.requires_grad_(True)
        du = du.to(device)
        pinn_l = pinn_loss(model(d), d, du, args)
        pinn_l_s.append(pinn_l.to('cpu').detach().numpy())
    pinn_l_s = np.concatenate(pinn_l_s, axis=0)**2
    # print(pinn_l_s.shape)
    mean = pinn_l_s.mean()
    std = pinn_l_s.std()
    if std > threshold * mean:
        print('Yeah!')
        wmax = torch.max(w)
        w[pinn_l_s > mean] += wmax
        w = w / torch.sum(w)
    # print(w)


def build_rightside(outputs, data, dU, args):

    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    with torch.no_grad():
        gradients = torch.autograd.grad(outputs=outputs, inputs=data,
                                        grad_outputs=torch.ones_like(outputs),
                                        create_graph=False, retain_graph=False)[0]
        grad_x = gradients[:, :ndim]
        grad_v = gradients[:, ndim:]

        '''
        res_q = -outputs-alpha_t*(data[:,ndim:]*grad_x-dU*grad_v)
        res_dq =-(1-gamma*alpha_t)*grad_v*kbt
        '''
        # print(outputs.shape,torch.sum((data[:,ndim:]*grad_x-dU*grad_v),dim=1,keepdim=True).shape)
        # print(torch.sum((data[:,ndim:]*grad_x-dU*grad_v),dim=1,keepdim=True).shape)
        res_q = -lam * outputs - \
            torch.sum((data[:, ndim:] * grad_x - dU * grad_v),
                      dim=1, keepdim=True)
        res_dq = (gamma - omega) * grad_v*kbt
        res_dqx = -eta * grad_x

    return res_q, res_dq, res_dqx


def pinn_loss(outputs, data, dU, args, create_graph=False):
    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=data,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True)[0]
    grad_x = grad[:, :ndim]
    grad_v = grad[:, ndim:]
    x = data[:, :ndim]
    v = data[:, ndim:]

    # \Delta q_v
    lap_v = torch.zeros(
        size=(
            outputs.shape[0],
        ),
        dtype=torch.float32,
        device=grad.device)
    for i in range(ndim):
        temp = torch.autograd.grad(outputs=grad[:,
                                                ndim + i],
                                   inputs=data,
                                   grad_outputs=torch.ones_like(grad[:,
                                                                     0]),
                                   create_graph=create_graph,
                                   retain_graph=True)[0]
        # print(temp[:,ndim+i].shape,lap_v.shape)
        lap_v += temp[:, ndim + i]
    lap_v.unsqueeze_(dim=1)

    return torch.sum((data[:, ndim:] * grad_x - (dU + gamma * v)
                     * grad_v), dim=1, keepdim=True) + kbt * gamma * lap_v


def split(dataset, batchsize, shuffle=True):
    length = len(dataset[0])
    if shuffle:
        per = torch.randperm(length)

        for i in range(len(dataset)):
            dataset[i] = dataset[i][per]

    batches = []
    N = int(np.ceil(length / batchsize))
    # print(N)
    for i in range(N):
        batches.append(
            [d[i * batchsize:min((i + 1) * batchsize, length)] for d in dataset])
    return iter(batches)


def train_step(model, model_o, dataset, batchsize, data_b, label_b, alpha_b,
               opt, num_epoches, device, args, alpha_l2=0, check_point=10):
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    for i in range(num_epoches):
        dataloader = split(dataset, batchsize, shuffle=True)
        for d, w, dU in dataloader:

            # torch.cuda.empty_cache()
            d = d.to(device)
            d.requires_grad_(True)
            dU = dU.to(device)
            w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            # print(d.shape,dU.shape)
            y = model(d)
            yy = model_o(d)

            # with torch.no_grad():
            #    print(torch.sum((y-yy)**2))
            res_q, res_dq, res_dqx = build_rightside(yy, d, dU, args)
            opt.zero_grad()
            # y = model(d)
            y_b = model(data_b)

            loss = loss_fn(y, d, w, res_q, res_dq, res_dqx, args)

            b_loss = b_lossfn(y_b, label_b)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2

            tot_loss = loss + b_loss * alpha_b + l2_reg * alpha_l2
            tot_loss.backward()
            opt.step()

        if i % check_point == 0:
            # print(i)
            # print(f"{i+1} epoches completed!")
            loss_list.append(loss.item())
            b_loss_list.append(b_loss.item())
            tot_loss_list.append(tot_loss.item())

            pinn_loss_list.append(
                torch.sum(
                    pinn_loss(
                        model(d),
                        d,
                        dU,
                        args)**2 * w).item()**0.5)

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


def train(model, data: torch.Tensor, w, batchsize, data_b, label_b, dU,
          alpha_b, lr, num_tsteps, num_epoches, device, args, checkpoint=10, threshold=1, alpha_l2=0, adaptive=True):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.detach()

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")
        model_o = copy.deepcopy(model)
        # y = model(data)
        # res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model, model_o, [data, w, dU], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint, alpha_l2=alpha_l2)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        if adaptive:
            weight_update(
                model,
                data,
                w,
                dU,
                batchsize,
                args,
                device,
                threshold=threshold)

        torch.cuda.empty_cache()
        print(f"itr{t}: Training completed!")

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


if __name__ == '__main__':
    pass
