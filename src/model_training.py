import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from muller_potential import MullerPotential

def wait():
    for i in range(10**100):
        print(f'Waiting {i}!')

def loss_fn(outputs, data, w, res_q, res_dq, res_dqx,
            args, have_rightside=False, rightside=0):

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
    if have_rightside:
        if rightside.shape != (rightside.shape[0], 1):
            rightside = rightside.unsqueeze(1)
        loss += torch.sum(w * outputs * rightside)

    # loss = loss / data.shape[0]
    # print(loss)

    return loss


b_lossfn = torch.nn.MSELoss()


def weight_update(model, data, w, dU, batchsize, args, device, threshold=1):
    dataset = [data, dU]
    batches = split(dataset, batchsize, shuffle=False)
    pinn_l_s = []
    for d, du in batches:
        # d = d.to(device)
        d.requires_grad_(True)
        # du = du.to(device)
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

def weight_update_1(model,data,w,dU,batchsize,args,device,xdim,vdim,beta = 1,N = 10,pinn_weight = 0.9, grad_weight = 0.05):
    sum = torch.zeros(size=(data.shape[0],1),dtype=torch.float32,device=device)
    sum_pinn = torch.zeros(size=(data.shape[0],1),dtype=torch.float32,device=device)
    for param in model.parameters():
        param.requires_grad = False
    for i in range(N):
        data[:,
             xdim:xdim + vdim] = torch.randn(size=(data.shape[0],
                                                   vdim),
                                             dtype=torch.float32,
                                             device=device) * np.sqrt(args['kbt'])
        dataset = [data, dU]
        batches = split(dataset, batchsize, shuffle=False)
        k = 0
        for d, du in batches:
            d = d.to(device)
            d.requires_grad_(True)
            du = du.to(device)
            y = model(d)

            with torch.no_grad():
                gradients = torch.autograd.grad(outputs=y, inputs=d,
                                                grad_outputs=torch.ones_like(
                                                    y),
                                                create_graph=False, retain_graph=True)[0]
                sum[k * batchsize:k * batchsize + d.shape[0]
                    ] += torch.sum(gradients**2, dim=1, keepdim=True)
            sum_pinn[k * batchsize:k * batchsize + d.shape[0]
                     ] += pinn_loss(model(d), d, du, args).detach()**2
            k+=1
            torch.cuda.empty_cache()
    sum = (sum / N)**beta
    sum = sum / torch.sum(sum)
    # print(pinn_l_s.shape)
    # w = alpha_beta*sum + (1-alpha_beta)*torch.ones(size=(data.shape[0],1),dtype=torch.float32,device=device)/data.shape[0]
    num_topk = 0.2
    num_topk = int(data.shape[0] * num_topk)
    topk_indices = torch.topk(sum_pinn.flatten(), num_topk, largest=True).indices
    w[topk_indices,:] *= 2
    w[:] = w/ torch.sum(w)
    w[:] = pinn_weight*w + grad_weight*sum + (1-pinn_weight-grad_weight)*torch.ones(size=(data.shape[0],1),dtype=torch.float32,device=device)/data.shape[0]
    for param in model.parameters():
        param.requires_grad = True
    return w


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

        # print(data[:, ndim:].shape, grad_x.shape, dU.shape, grad_v.shape,(data[:, ndim:] * grad_x).shape, (dU * grad_v).shape)
        res_q = -lam * outputs - \
            torch.sum((data[:, ndim:] * grad_x - dU * grad_v),
                      dim=1, keepdim=True)
        res_dq = (gamma - omega) * grad_v * kbt
        res_dqx = -eta * grad_x

    return res_q, res_dq, res_dqx

def pinn_loss_grad(grad_x,grad_v, data, dU, args, create_graph=False):
    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    x = data[:, :ndim]
    v = data[:, ndim:]

    # \Delta q_v
    lap_v = torch.zeros(
        size=(
            grad_x.shape[0],
        ),
        dtype=torch.float32,
        device=grad_x.device)
    for i in range(ndim):
        temp = torch.autograd.grad(outputs=grad_v[:,i],
                                   inputs=data,
                                   grad_outputs=torch.ones_like(grad_v[:,
                                                                     0]),
                                   create_graph=create_graph,
                                   retain_graph=True)[0]
        # print(temp[:,ndim+i].shape,lap_v.shape)
        lap_v += temp[:, ndim + i]
    lap_v.unsqueeze_(dim=1)

    ttt = torch.sum((data[:, ndim:] * grad_x - (dU + gamma * v)
                     * grad_v), dim=1, keepdim=True)
    return ttt + kbt * gamma * lap_v

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
    if not create_graph:
        #with torch.no_grad():
        return pinn_loss_grad(grad_x, grad_v, data, dU, args, create_graph=create_graph)
    else:
        return pinn_loss_grad(grad_x, grad_v, data, dU, args, create_graph=create_graph)
    



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
               opt, num_epoches, device, args, alpha_l2=0, check_point=10, have_rightside=False):
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    
    for i in range(num_epoches):
        with torch.no_grad():
            dataloader = split(dataset, batchsize, shuffle=True)
        for batch in dataloader:
            
            d = batch[0]
            w = batch[1]
            dU = batch[2]
            if have_rightside:
                rightside = batch[3]
            else:
                rightside = 0
                #rightside = 0
            
            
            # torch.cuda.empty_cache()
            # d=d.to(device)
            d.requires_grad_(True)
            # dU = dU.to(device)
            # w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            # print(d.shape,dU.shape)
            
            yy = model_o(d)
            # with torch.no_grad():
            #    print(torch.sum((y-yy)**2))
            res_q, res_dq, res_dqx = build_rightside(yy, d, dU, args)
            opt.zero_grad()
            # y = model(d)
            y = model(d)
            y_b = model(data_b)
            
            loss = loss_fn(
                y,
                d,
                w,
                res_q,
                res_dq,
                res_dqx,
                args,
                have_rightside=have_rightside,
                rightside=rightside)
            
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


def train_pinn(model, data: torch.Tensor,w, batchsize, data_b, label_b, dU,
          alpha_b, lr, num_tsteps, num_epoches, device, args,xdim,vdim, checkpoint=10,threshold=1,alpha_l2=0,adaptive=True,beta=1,alpha_beta=1,valid_checkpoint = -1,valid_data = None,valid_w=None,valid_dU = None,pinn_weight=0.9, grad_weight=0.05):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
    dU = dU.to(device)
    data.requires_grad_(False)

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    valid_pinn_loss = []
    if valid_data is not None and valid_dU is not None:
        valid_data = valid_data.to(device)
        valid_data.requires_grad_(True)
        valid_dU = valid_dU.to(device)
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")
        # data[:,xdim:xdim+vdim] = torch.randn(size=(data.shape[0],vdim),dtype=torch.float32,device=device) * np.sqrt(args['kbt'])
        # y = model(data)
        # res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        model.train()
        dataset = [data, w, dU]
        batches = split(dataset, batchsize, shuffle=True)
        for d, w, du in batches:
            # d = d.to(device)
            d.requires_grad_(True)
            # dU = dU.to(device)
            # w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            opt.zero_grad()
            # y = model(d)
            y = model(d)
            y_b = model(data_b)

            loss = torch.sum(
                pinn_loss(
                    y,
                    d,
                    du,
                    args,
                    create_graph=True)**2 *
                w)

            b_loss = b_lossfn(y_b, label_b)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2

            tot_loss = loss + b_loss * alpha_b + l2_reg * alpha_l2
            tot_loss.backward()
            opt.step()
            loss_list.append(loss.item())
            b_loss_list.append(b_loss.item())
            tot_loss_list.append(tot_loss.item())

        if adaptive and t % 20 == 0:
            w[:] = weight_update_1(model, data, w, dU, batchsize, args, device, xdim, vdim, beta=beta,pinn_weight=pinn_weight, grad_weight=grad_weight)

        if valid_checkpoint > 0 and t % valid_checkpoint == 0:

            for param in model.parameters():
                param.requires_grad = False
            if valid_data is not None and valid_dU is not None:
                y = model(valid_data)
                if valid_w is not None:
                    valid_w = valid_w.to(device)
                else:
                    valid_w = torch.ones(
                        size=(
                            valid_data.shape[0],
                            1),
                        dtype=torch.float32,
                        device=device)
                    valid_w = valid_w / torch.sum(valid_w)
            pinn_l = pinn_loss(y, valid_data, valid_dU, args)
            pinn_l = torch.sum(pinn_l**2 * valid_w).item()**0.5
            print(f"Validation at step {t}: PINN Loss: {pinn_l:.4f}")
            valid_pinn_loss.append(pinn_l)
            for param in model.parameters():
                param.requires_grad = True

        torch.cuda.empty_cache()
        print(f"itr{t}: Training completed!")

    if valid_checkpoint > 0:
        return loss_list, b_loss_list, tot_loss_list, valid_pinn_loss
    else:
        return loss_list, b_loss_list, tot_loss_list


def train_resample(model, data: torch.Tensor, w, batchsize, data_b, label_b, dU,
                   alpha_b, lr, num_tsteps, num_epoches, device, args, xdim, vdim, checkpoint=10, threshold=1, alpha_l2=0, adaptive=True, beta=1, alpha_beta=1, valid_checkpoint=-1, valid_data=None, valid_w=None, valid_dU=None, pinn_weight=0.9, grad_weight=0.05):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
    dU = dU.to(device)

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    valid_pinn_loss = []
    if valid_data is not None and valid_dU is not None:
        valid_data = valid_data.to(device)
        valid_data.requires_grad_(True)
        valid_dU = valid_dU.to(device)
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")

        data[:,
             xdim:xdim + vdim] = torch.randn(size=(data.shape[0],
                                                   vdim),
                                             dtype=torch.float32,
                                             device=device) * np.sqrt(args['kbt'])
        model_o = copy.deepcopy(model)
        for param in model_o.parameters():
            param.requires_grad = False
        # y = model(data)
        # res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model, model_o, [data, w, dU], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint, alpha_l2=alpha_l2, have_rightside=False)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        if adaptive and t % 20 == 0:
            w[:] = weight_update_1(model, data, w, dU, batchsize, args, device, xdim, vdim, beta=beta,pinn_weight=pinn_weight, grad_weight=grad_weight)

        if valid_checkpoint > 0 and t % valid_checkpoint == 0:

            for param in model.parameters():
                param.requires_grad = False
            if valid_data is not None and valid_dU is not None:
                y = model(valid_data)
                if valid_w is not None:
                    valid_w = valid_w.to(device)
                else:
                    valid_w = torch.ones(
                        size=(
                            valid_data.shape[0],
                            1),
                        dtype=torch.float32,
                        device=device)
                    valid_w = valid_w / torch.sum(valid_w)
            pinn_l = pinn_loss(y, valid_data, valid_dU, args).detach()
            pinn_l = torch.sum(pinn_l**2 * valid_w).item()**0.5
            print(f"Validation at step {t}: PINN Loss: {pinn_l:.4f}")
            valid_pinn_loss.append(pinn_l)
            for param in model.parameters():
                param.requires_grad = True
        torch.cuda.empty_cache()
        

    if valid_checkpoint > 0:
        return loss_list, b_loss_list, tot_loss_list, pinn_loss_list, valid_pinn_loss
    else:
        return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


def train_resample_qref(model, data: torch.Tensor, w, q_ref, batchsize, data_b, label_b, dU,
                        alpha_b, lr, num_tsteps, num_epoches, device, args, xdim, vdim, checkpoint=10, threshold=1, alpha_l2=0, adaptive=True, beta=1, alpha_beta=1, valid_checkpoint=-1, valid_data=None, valid_w=None, valid_dU=None, pinn_weight=0.9, grad_weight=0.05):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
    dU = dU.to(device)
    q_ref.to(device)
    for param in q_ref.parameters():
        param.requires_grad = False
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    valid_pinn_loss = []
    if valid_data is not None and valid_dU is not None:
        valid_data = valid_data.to(device)
        valid_data.requires_grad_(True)
        valid_dU = valid_dU.to(device)

   
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")

        data[:,
             xdim:xdim + vdim] = torch.randn(size=(data.shape[0],
                                                   vdim),
                                             dtype=torch.float32,
                                             device=device) * np.sqrt(args['kbt'])
        model_o = copy.deepcopy(model)
        data.requires_grad_(True)
        rightside = pinn_loss(q_ref(data), data, dU, args).detach()
        torch.cuda.empty_cache()
        data.requires_grad_(False)
        
        for param in model_o.parameters():
            param.requires_grad = False

        # y = model(data)
        # res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model, model_o, [data, w, dU,rightside], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint, alpha_l2=alpha_l2, have_rightside=True)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        print(f"itr{t}: Training completed!")
        if adaptive and t % 5 == 0:
            w[:] = weight_update_1(
                model,
                data,
                w,
                dU,
                batchsize,
                args,
                device,
                xdim,
                vdim,
                beta=beta,
                pinn_weight=pinn_weight,
                grad_weight=grad_weight)
            print(f"itr{t}: Resampling completed!")

        if valid_checkpoint > 0 and t % valid_checkpoint == 0:

            for param in model.parameters():
                param.requires_grad = False
            if valid_data is not None and valid_dU is not None:
                y = model(valid_data)
                if valid_w is not None:
                    valid_w = valid_w.to(device)
                else:
                    valid_w = torch.ones(
                        size=(
                            valid_data.shape[0],
                            1),
                        dtype=torch.float32,
                        device=device)
                    valid_w = valid_w / torch.sum(valid_w)
            pinn_l = pinn_loss(y, valid_data, valid_dU, args).detahch()
            pinn_l = torch.sum(pinn_l**2 * valid_w).item()**0.5
            print(f"Validation at step {t}: PINN Loss: {pinn_l:.4f}")
            valid_pinn_loss.append(pinn_l)
            for param in model.parameters():
                param.requires_grad = True
        torch.cuda.empty_cache()
        

    if valid_checkpoint > 0:
        return loss_list, b_loss_list, tot_loss_list, pinn_loss_list, valid_pinn_loss
    else:
        return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


if __name__ == '__main__':
    ndim = 2
    ndata = 1000
    xdim = ndim
    vdim = ndim
    args = {
        'ndim': ndim,
        'kbt': 1.0,
        'omega': 1.0,
        'lam': 1.0,
        'eta': 1.0,
        'gamma': 1.0
    }
    data = torch.randn(size=(ndata, xdim + vdim), dtype=torch.float32)
    data.requires_grad_(True)

    y = torch.sum(data[:, xdim:]**4, dim=1, keepdim=True)
    dU = torch.zeros_like(data[:, :xdim])
    print(pinn_loss(y, data, dU, args) - 12 *
          torch.sum(data[:, xdim:]**2, dim=1, keepdim=True))
