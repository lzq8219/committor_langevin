import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt


def wait():
    for i in range(10**100):
        print(f'Waiting {i}!')


class MyDataset(Dataset):
    def __init__(self, data,w,dU,mass=None):
        self.data = data
        self.w = w
        self.dU = dU
        if mass is None:
            self.mass = torch.ones(size=(1, data.shape[1]//2), dtype=torch.float32, device=data.device)
        else:
            self.mass = mass

    def __len__(self):
        return len(self.data)
    
    def v_resample(self,xdim,vdim,kbt):
        self.data[:,xdim:xdim+vdim] = torch.randn(size=(self.data.shape[0],vdim),dtype=torch.float32,device=self.data.device) * np.sqrt(kbt)/torch.sqrt(self.mass)

    def __getitem__(self, idx):
        return self.data[idx,:], self.w[idx,:],self.dU[idx,:]
    
    def get_mass(self):
        return self.mass    
    def weight_update(self, new_w):
        self.w = new_w

def loss_fn(outputs, data, w, res_q, res_dq, res_dqx,
            args,mass = None, have_rightside=False, rightside=0):

    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    grad_mask = data.requires_grad
    data.requires_grad_(True)
    gradients = torch.autograd.grad(outputs=outputs, inputs=data,
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
    grad_x = gradients[:, :ndim]
    grad_v = gradients[:, ndim:]
    data.requires_grad_(False)
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

    if mass is None:
        mass = torch.ones(size=(1, ndim), dtype=torch.float32, device=data.device) 
    # print(grad_v.shape,res_dq.shape,w.shape)
    # print(outputs.shape,res_q.shape,w.shape)

    if mass.shape != (1,mass.shape[0]):
        mass = mass.unsqueeze(0)
    loss = (omega * kbt * torch.sum(w * grad_v**2/mass) + lam * torch.sum(w * outputs**2) + eta * torch.sum(w * grad_x**2/mass)) / 2 \
        + torch.sum(w * grad_v * res_dq/mass) + torch.sum(w * outputs *
                                                    res_q) + torch.sum(w * grad_x * res_dqx/mass)
    if have_rightside:
        if rightside.shape != (rightside.shape[0], 1):
            rightside = rightside.unsqueeze(1)
        loss += torch.sum(w * outputs * rightside)

    # loss = loss / data.shape[0]
    # print(loss)
    data.requires_grad_(grad_mask)
    return loss

def _loss_HSS_2_pinn(q,grad_x,grad_v, data, dU, args,rightside_HSS_2, create_graph=False,mass = None):
    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']
    x = data[:, :ndim]
    v = data[:, ndim:]
    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=data.device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)
    Aq = torch.sum(v* grad_x - dU*grad_v/mass, dim=1, keepdim=True)
    left_side = lam * q - Aq
    return left_side - rightside_HSS_2

def _right_side_HSS_2(q,grad_v, data, dU, args,mass = None):
    ndim = args['ndim']
    kbt = args['kbt']
    gamma = args['gamma']
    lam = args['lam']
    v = data[:, ndim:]

    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=data.device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)

    lap_v = torch.zeros(
        size=(
            grad_v.shape[0],
        ),
        dtype=torch.float32,
        device=grad_v.device)
    for i in range(ndim):
        temp = torch.autograd.grad(outputs=grad_v[:,i],
                                   inputs=data,
                                   grad_outputs=torch.ones_like(grad_v[:,
                                                                     0]),
                                   create_graph=False,
                                   retain_graph=True)[0]
        # print(temp[:,ndim+i].shape,lap_v.shape)
        lap_v += temp[:, ndim + i]/ mass[0, i]
    lap_v.unsqueeze_(dim=1)


    right_side_HSS_2 = lam*q+ kbt * gamma * lap_v - gamma *torch.sum(v * grad_v, dim=1, keepdim=True)
    return right_side_HSS_2.detach()

def _leap_frog_step(data, grad_fn, args,mass = None,dt = 0.001):
    ndim = args['ndim']
    x = data[:, :ndim]
    v = data[:, ndim:]

    if mass is None:
        mass = torch.ones(size=(1, data.shape[1]//2), dtype=torch.float32, device=data.device)
    if mass.shape != (1, data.shape[1]//2):
        mass = mass.unsqueeze(0)

    with torch.no_grad():
        dU = grad_fn(x)
        v_half = v + 0.5 * dt * (-dU/mass )
        x_new = x + dt * v_half
        dU_new = grad_fn(x_new)
        v_new = v_half + 0.5 * dt * (-dU_new/mass)
    new_data = torch.cat((x_new, v_new), dim=1)
    return new_data
def _loss_HSS_2_char(q,q_new,right_side,right_side_new, args,mass = None,dt = 0.001):
    dqdt = (q_new - q) / dt
    r = (right_side+right_side_new)/2
    qq = (q + q_new) / 2
    return dqdt + qq -r


def _grad_x_v(y,d,ndim,create_graph=False,retain_graph=False):
    grad = torch.autograd.grad(outputs=y, inputs=d,
                                grad_outputs=torch.ones_like(
                                    y),
                                create_graph=create_graph, retain_graph=retain_graph)[0]
    return grad[:,:ndim],grad[:,ndim:]


    

def loss_fn_HSS_2(outputs, data, w, res_q, res_dq, res_dqx,
            args,mass = None, have_rightside=False, rightside=0):
    pass

def loss_fn_overdamped(outputs, data, w, 
            args,mass = None):

    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    gradients = torch.autograd.grad(outputs=outputs, inputs=data,
                                    grad_outputs=torch.ones_like(outputs),
                                    create_graph=True, retain_graph=True)[0]
    
    '''
    loss = (kbt*torch.sum(grad_v**2)+torch.sum(outputs**2))/2 \
            +torch.sum(grad_v*res_dq)+torch.sum(outputs*res_q)
    '''
    # print(grad_v.shape,res_dq.shape)
    # print(outputs.shape,res_q.shape)
    if w.shape != (w.shape[0], 1):
        w = w.unsqueeze(1)

    if mass is None:
        mass = torch.ones(size=(1, ndim), dtype=torch.float32, device=data.device) 
    # print(grad_v.shape,res_dq.shape,w.shape)
    # print(outputs.shape,res_q.shape,w.shape)

    if mass.shape != (1,mass.shape[0]):
        mass = mass.unsqueeze(0)
    loss =  torch.sum(w * gradients**2/mass) 
    

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

def weight_update_1(model,data,w,dU,batchsize,args,device,xdim,vdim,beta = 1,N = 10,pinn_weight = 0.9, grad_weight = 0.05,mass=None):
    sum = torch.zeros(size=(data.shape[0],1),dtype=torch.float32,device=device)
    sum_pinn = torch.zeros(size=(data.shape[0],1),dtype=torch.float32,device=device)
    if mass is None:    
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=device)
    for param in model.parameters():
        param.requires_grad = False

    mass2 = torch.cat((mass, mass), dim=1)
    print(mass2.shape)
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
                    ] += torch.sum(gradients**2/mass2, dim=1, keepdim=True)
            sum_pinn[k * batchsize:k * batchsize + d.shape[0]
                     ] += pinn_loss(model(d), d, du, args,mass=mass).detach()**2
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


def build_rightside(outputs, data, dU, args,mass = None):

    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1] ), dtype=torch.float32, device=data.device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)

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
            torch.sum((data[:, ndim:] * grad_x - dU/mass * grad_v),
                      dim=1, keepdim=True)
        res_dq = (gamma - omega) * grad_v * kbt
        res_dqx = -eta * grad_x

    return res_q.detach(), res_dq.detach(), res_dqx.detach()

def pinn_loss_grad(grad_x,grad_v, data, dU, args, create_graph=False,mass = None):
    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=data.device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)

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
        lap_v += temp[:, ndim + i]/ mass[0, i]
    lap_v.unsqueeze_(dim=1)

    ttt = torch.sum((data[:, ndim:] * grad_x - (dU/mass + gamma * v)
                     * grad_v), dim=1, keepdim=True)
    return ttt + kbt * gamma * lap_v

def pinn_loss(outputs, data, dU, args, create_graph=False,mass = None):
    ndim = args['ndim']
    kbt = args['kbt']
    omega = args['omega']
    lam = args['lam']
    eta = args['eta']
    gamma = args['gamma']

    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=data.device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)

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
        return pinn_loss_grad(grad_x, grad_v, data, dU, args, create_graph=create_graph,mass=mass)
    else:
        return pinn_loss_grad(grad_x, grad_v, data, dU, args, create_graph=create_graph,mass=mass)
    



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
               opt, num_epoches, device, args, alpha_l2=0,alpha_char=0, check_point=10, have_rightside=False,mass = None):
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    
    if mass is None:
        mass = torch.ones(size=(1, args['ndim']), dtype=torch.float32, device=device)
    if mass.shape != (1, args['ndim']):
        mass = mass.unsqueeze(0)
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

            
            yy = model_o(d)
            res_q, res_dq, res_dqx = build_rightside(yy, d, dU, args,mass=mass)
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
                rightside=rightside,
                mass=mass)
            
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

def train_step_HSS_2(model, model_o, dataset, batchsize, data_b, label_b, alpha_b,
               opt, num_epoches, device, args, alpha_l2=0, alpha_char=0, check_point=10,mass = None,dt = 0.001):
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list,char_loss = [], [], [], [], []
    if mass is None:
        mass = torch.ones(size=(1, args['ndim']), dtype=torch.float32, device=device)
    if mass.shape != (1, args['ndim']):
        mass = mass.unsqueeze(0)
    for i in range(num_epoches):
        with torch.no_grad():
            dataloader = split(dataset, batchsize, shuffle=True)
        for batch in dataloader:
            
            d = batch[0]
            w = batch[1]
            dU = batch[2]
            d_new = batch[3]
            dU_new = batch[4]
            r = batch[5]
            r_new = batch[6]
            
                #rightside = 0
            
            
            # torch.cuda.empty_cache()
            # d=d.to(device)
            d.requires_grad_(True)
            d_new.requires_grad_(True)

            # dU = dU.to(device)
            # w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            # print(d.shape,dU.shape)
            
            y = model(d)
            grad_x,grad_v = _grad_x_v(y,d,args['ndim'],create_graph=True,retain_graph=True)
            d.requires_grad_(False)
            d_new.requires_grad_(False)
            y_new = model(d_new)
            
            opt.zero_grad()
            # y = model(d)
            loss_pinn = torch.sum(_loss_HSS_2_pinn(y,grad_x,grad_v,d,dU,args,r,mass=mass)**2*w)
            loss_char = torch.sum(_loss_HSS_2_char(y,y_new,r,r_new,args,mass=mass)**2*w)
            y = model(d)
            y_b = model(data_b)
            b_loss = b_lossfn(y_b, label_b)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2

            tot_loss = loss_pinn + alpha_char*loss_char + b_loss * alpha_b + l2_reg * alpha_l2
            
            tot_loss.backward()
            opt.step()
            

        if i % check_point == 0:
            # print(i)
            # print(f"{i+1} epoches completed!")
            loss_list.append(loss_pinn.item())
            b_loss_list.append(b_loss.item())
            tot_loss_list.append(tot_loss.item())
            d.requires_grad_(True)
            pinn_loss_list.append(
                torch.sum(
                    pinn_loss(
                        model(d),
                        d,
                        dU,
                        args)**2 * w).item()**0.5)
            char_loss.append(
                torch.sum(loss_char).item()**0.5)

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list, char_loss


def train_overdamped(model, data: torch.Tensor,w, batchsize, data_b, label_b,
          alpha_b, lr, num_tsteps, num_epoches, device, args,xdim,vdim,
           mass = None, checkpoint=10,alpha_l2=0,valid_checkpoint = -1,valid_data = None,valid_w=None,valid_dU = None,model_old = None):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
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
        dataset = [data, w]
        batches = split(dataset, batchsize, shuffle=True)
        for d, w in batches:
            # d = d.to(device)
            d.requires_grad_(True)
            
            # dU = dU.to(device)
            # w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            opt.zero_grad()
            # y = model(d)
            if model_old is not None:
                y = model(d)- model_old(d)
                y_b = model(data_b) - model_old(data_b)
            else:
                y = model(d)
                y_b = model(data_b)
            loss = loss_fn_overdamped(
                y,
                d,
                w,
                args,
                mass=mass)

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
    
def train_HSS(model, data: torch.Tensor, w, batchsize, data_b, label_b, dU,
                   alpha_b, lr, num_tsteps, num_epoches, device, args, xdim, vdim, 
                   checkpoint=10, alpha_l2=0, adaptive=True,
                   valid_checkpoint=-1, valid_data=None, valid_w=None, valid_dU=None,
                    alpha_char=1,lr_HSS_2=1e-3,num_epoches_HSS_2=100,dU_func=None,dt=0.001):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
    dU = dU.to(device)

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    loss_list_2, b_loss_list_2, tot_loss_list_2, pinn_loss_list_2 = [], [], [], []
    char_loss_list = []
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
        
        data_new = _leap_frog_step(data,dU_func,args,dt=dt)
        dU_new = dU_func(data_new)
        model_o = copy.deepcopy(model)
        for param in model_o.parameters():
            param.requires_grad = False
        
        
        opt = optim.Adam(model.parameters(), lr=lr)
        
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model, model_o, [data, w, dU], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint, alpha_l2=alpha_l2, have_rightside=False)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl

        # HSS second step
        model_o = copy.deepcopy(model)
        for param in model_o.parameters():
            param.requires_grad = False
        opt = optim.Adam(model.parameters(), lr=lr_HSS_2)
        def build_rightside_HSS_2(q_model,d):
            mask_grad = d.requires_grad
            d.requires_grad_(True)
            y = q_model(d)             
            grad_v = _grad_x_v(y,d,args['ndim'],create_graph=True,retain_graph=True)[1]
            r = _right_side_HSS_2(y,grad_v,d,dU,args)
            d.requires_grad_(mask_grad)
            return r.detach()
        r = build_rightside_HSS_2(model_o,data)
        r_new = build_rightside_HSS_2(model_o,data_new)
        dataset = [data, w, dU, data_new, dU_new, r, r_new]
        ll, bl, tl, pl,cl = train_step_HSS_2(model, model_o, dataset, batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches_HSS_2, device, args,
                                      check_point=checkpoint, alpha_l2=0,alpha_char=alpha_char, mass=None, dt=dt)
        loss_list_2 += ll
        b_loss_list_2 += bl
        tot_loss_list_2 += tl
        pinn_loss_list_2 += pl
        char_loss_list += cl
        del model_o, r, r_new, dataset
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
        
    loss_1 = [loss_list, b_loss_list, tot_loss_list, pinn_loss_list]
    loss_2 = [loss_list_2, b_loss_list_2, tot_loss_list_2, pinn_loss_list_2,char_loss_list]
    if valid_checkpoint > 0:
        return loss_1,loss_2, valid_pinn_loss
    else:
        return loss_1,loss_2, valid_pinn_loss

    


def train_mass(model, data: torch.Tensor, w, batchsize, data_b, label_b, dU,
                   alpha_b, lr, num_tsteps, num_epoches, device, args, xdim, vdim, 
                   mass = None, checkpoint=10, threshold=1, alpha_l2=0, adaptive=True, beta=1, alpha_beta=1, valid_checkpoint=-1, valid_data=None, valid_w=None, valid_dU=None, pinn_weight=0.9, grad_weight=0.05):
    torch.cuda.empty_cache()
    # data = data.to(device)
    # dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.to(device)
    dU = dU.to(device)
    if mass is None:
        mass = torch.ones(size=(1, dU.shape[1]), dtype=torch.float32, device=device)
    if mass.shape != (1, dU.shape[1]):
        mass = mass.unsqueeze(0)

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
                                             device=device) * np.sqrt(args['kbt'])/torch.sqrt(mass)
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
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint, alpha_l2=alpha_l2, have_rightside=False,mass=mass)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        # if adaptive and t % 20 == 0:
        #    w[:] = weight_update_1(model, data, w, dU, batchsize, args, device, xdim, vdim, beta=beta,pinn_weight=pinn_weight, grad_weight=grad_weight,mass = mass)

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
            pinn_l = pinn_loss(y, valid_data, valid_dU, args,mass=mass).detach()
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
