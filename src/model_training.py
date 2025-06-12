import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from muller_potential import MullerPotential


def loss_fn(outputs, data,w, res_q, res_dq, res_dqx, args):

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

    #print(grad_v.shape,res_dq.shape,w.shape)
    #print(outputs.shape,res_q.shape,w.shape)
    loss = (omega * kbt * torch.sum(w*grad_v**2) + lam * torch.sum(w*outputs**2) + eta * torch.sum(w*grad_x**2)) / 2 \
        + torch.sum(w*grad_v * res_dq) + torch.sum(w*outputs *
                                                 res_q) + torch.sum(w*grad_x * res_dqx)

    # loss = loss / data.shape[0]
    #print(loss)

    return loss


b_lossfn = torch.nn.MSELoss()

def weight_update(model,data,w,dU,batchsize,args,device,threshold = 1):
    dataset = [data,dU]
    batches = split(dataset,batchsize,shuffle=False)
    pinn_l_s = []
    for d, du in batches:
        #d = d.to(device)
        d.requires_grad_(True)
        #du = du.to(device)
        pinn_l = pinn_loss(model(d),d,du,args)
        pinn_l_s.append(pinn_l.to('cpu').detach().numpy())
    pinn_l_s = np.concatenate(pinn_l_s,axis=0)**2
    # print(pinn_l_s.shape)
    mean = pinn_l_s.mean()
    std = pinn_l_s.std()
    if std>threshold * mean:
        print('Yeah!')
        wmax = torch.max(w)
        w[pinn_l_s>mean] +=wmax
        w = w/torch.sum(w)
    # print(w)

def weight_update_1(model,data,w,dU,batchsize,args,device,xdim,vdim,threshold = 1,beta = 1,N = 10,alpha_beta = 1):
    sum = torch.zeros(size=(data.shape[0],1),dtype=torch.float32,device=device)
    for i in range(N):
        data[:,xdim:xdim+vdim] = torch.randn(size=(data.shape[0],vdim),dtype=torch.float32,device=device) * np.sqrt(args['kbt'])
        dataset = [data,dU]
        batches = split(dataset,batchsize,shuffle=False)
        k=0
        for d, du in batches:
            d = d.to(device)
            d.requires_grad_(True)
            du = du.to(device)
            y = model(d)
            gradients = torch.autograd.grad(outputs=y, inputs=d,
                                            grad_outputs=torch.ones_like(y),
                                            create_graph=False, retain_graph=False)[0]
            sum[k*batchsize:k*batchsize+d.shape[0]]+=torch.sum(gradients**2,dim=1,keepdim=True)
    sum = (sum / N)**beta
    sum = sum/torch.sum(sum)
    # print(pinn_l_s.shape)
    w = alpha_beta*sum + (1-alpha_beta)*torch.ones(size=(data.shape[0],1),dtype=torch.float32,device=device)/data.shape[0]
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
    #print(N)
    for i in range(N):
        batches.append(
            [d[i * batchsize:min((i + 1) * batchsize, length)] for d in dataset])
    return iter(batches)


def train_step(model,model_o, dataset, batchsize, data_b, label_b, alpha_b,
               opt, num_epoches, device, args, alpha_l2=0, check_point=10):
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    for i in range(num_epoches):
        dataloader = split(dataset, batchsize, shuffle=True)
        for d,w, dU in dataloader:

            # torch.cuda.empty_cache()
            #d=d.to(device)
            d.requires_grad_(True)
            #dU = dU.to(device)
            #w = w.to(device)
            # rq=rq.to(device)
            # rdq=rdq.to(device)
            #print(d.shape,dU.shape)
            
            yy = model_o(d)
            # with torch.no_grad():
            #    print(torch.sum((y-yy)**2))
            res_q, res_dq, res_dqx = build_rightside(yy,d,dU,args)
            opt.zero_grad()
            # y = model(d)
            y = model(d)
            y_b = model(data_b)

            loss = loss_fn(y, d,w, res_q, res_dq, res_dqx, args)
            
            b_loss = b_lossfn(y_b, label_b)
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            
            tot_loss =  loss + b_loss * alpha_b + l2_reg * alpha_l2
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
                        args)**2*w).item()**0.5)

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


def train(model, data: torch.Tensor,w, batchsize, data_b, label_b, dU,
          alpha_b, lr, num_tsteps, num_epoches, device, args, checkpoint=10,threshold=1,alpha_l2=0,adaptive=True):
    torch.cuda.empty_cache()
    #data = data.to(device)
    #dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data = data.detach()
    
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")
        model_o = copy.deepcopy(model)
        # y = model(data)
        #res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model,model_o, [data,w,dU], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint,alpha_l2=alpha_l2)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        if adaptive:
            weight_update(model,data,w,dU,batchsize,args,device,threshold=threshold)

        torch.cuda.empty_cache()
        print(f"itr{t}: Training completed!")

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list



def train_resample(model, data: torch.Tensor,w, batchsize, data_b, label_b, dU,
          alpha_b, lr, num_tsteps, num_epoches, device, args,xdim,vdim, checkpoint=10,threshold=1,alpha_l2=0,adaptive=True,beta=1,alpha_beta=1):
    torch.cuda.empty_cache()
    #data = data.to(device)
    #dU = dU.to(device)
    label_b = label_b.to(device)
    data_b = data_b.to(device)
    data=data.to(device)
    dU = dU.to(device)
    
    
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = [], [], [], []
    for t in range(num_tsteps):
        print(f"itr{t}: Building dataset!")
        
        data[:,xdim:xdim+vdim] = torch.randn(size=(data.shape[0],vdim),dtype=torch.float32,device=device) * np.sqrt(args['kbt'])
        model_o = copy.deepcopy(model)
        
        # y = model(data)
        #res_q, res_dq, res_dqx = build_rightside(y, data, dU, args)
        # dataset = TensorDataset(data.to('cpu'),res_q.to('cpu'),res_dq.to('cpu'))
        opt = optim.Adam(model.parameters(), lr=lr)
        # dataloader = (data,res_q,res_dq)
        print(f"itr{t}: Training!")
        ll, bl, tl, pl = train_step(model,model_o, [data,w,dU], batchsize,
                                    data_b, label_b, alpha_b, opt, num_epoches, device, args, check_point=checkpoint,alpha_l2=alpha_l2)
        loss_list += ll
        b_loss_list += bl
        tot_loss_list += tl
        pinn_loss_list += pl
        if adaptive:
            w=weight_update_1(model,data,w,dU,batchsize,args,device,xdim,vdim,threshold=threshold,beta=beta,alpha_beta=alpha_beta)

        torch.cuda.empty_cache()
        print(f"itr{t}: Training completed!")

    return loss_list, b_loss_list, tot_loss_list, pinn_loss_list


if __name__ == '__main__':
    
    ndim = 2
    gamma = 100
    kbt = 10
    lam = 10
    eta = 10
    omega = gamma

    args = {
        "ndim": ndim,
        "gamma": gamma,
        "kbt": kbt,
        "lam": lam,
        "eta": eta,
        "omega": omega
    }

    # sample
    '''
    Nx_sample = 1000
    Nv_sample = 1000
    '''
    N_sample = 10**7
    NA = 5000
    NB = 5000

    batch_size = 2048  # not implement

    layers = [2 * ndim, 8, 64, 64, 8, 1]
    activ = 'sigmoid'

    alpha_t = 1
    T = 20
    Nt = int(T / alpha_t)
    Nsteps = 20
    lr = 1e-3

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    xmin, xmax = -1.5, 1.2
    ymin, ymax = -.2, 2
    x1 = (torch.rand(size=(N_sample, 1), dtype=torch.float32)) * \
        (xmax - xmin) + xmin
    x2 = (torch.rand(size=(N_sample, 1), dtype=torch.float32)) * \
        (ymax - ymin) + ymin
    x = torch.cat((x1, x2), dim=1)
    potential = MullerPotential()
    x = x[potential.potential(x)<0,:]
    N_sample = x.shape[0]
    print(N_sample)

    v = torch.randn(size=(N_sample, ndim), dtype=torch.float32) * np.sqrt(kbt)
    data = torch.cat((x, v), dim=1)
    dU = potential.gradient(data[:, :ndim])

    r = 0.2
    vA = torch.randn(size=(NA, ndim), dtype=torch.float32) * np.sqrt(kbt)
    xA = torch.from_numpy(potential.points_in_a(NA, r).astype(np.float32))
    xA = torch.cat((xA, vA), dim=1)
    vB = torch.randn(size=(NB, ndim), dtype=torch.float32) * np.sqrt(kbt)
    xB = torch.from_numpy(potential.points_in_b(NB, r).astype(np.float32))
    xB = torch.cat((xB, vB), dim=1)
    labelA = 0 * torch.ones_like(xA[:, 0])
    labelB = 1 * torch.ones_like(xB[:, 0])

    data_b = torch.cat((xA, xB), dim=0)
    label_b = torch.cat((labelA, labelB), dim=0).unsqueeze(dim=1)
    del xA, xB, labelA, labelB

    b_lossfn = torch.nn.MSELoss()
    b_lossfn(data[:, 0], torch.ones_like(data[:, 0]))

    q = FunctionModel(layer_sizes=layers, activation=activ)

    data.requires_grad_(True)
    w = torch.ones(size=(data.shape[0],1),dtype=torch.float32,device=device)
    w = w/torch.sum(w)
    q.to(device)
    batch_size = 2**21
    # eta = 10
    # lr = 1e-3
    # kbt = 1

    
    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = train(model=q,
                                                                  w=w,
                                                                  data=data,
                                                                  batchsize=batch_size,
                                                                  data_b=data_b,
                                                                  label_b=label_b,
                                                                  alpha_b=100,
                                                                  lr=1e-3,
                                                                  num_tsteps=10,
                                                                  num_epoches=Nsteps,
                                                                  args=args,
                                                                  device=device,
                                                                  dU=dU,
                                                                  checkpoint=10)

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = train(model=q,
                                                                  w=w,
                                                                  data=data,
                                                                  batchsize=batch_size,
                                                                  data_b=data_b,
                                                                  label_b=label_b,
                                                                  alpha_b=100,
                                                                  lr=1e-4,
                                                                  num_tsteps=10,
                                                                  num_epoches=Nsteps,
                                                                  args=args,
                                                                  device=device,
                                                                  dU=dU,
                                                                  checkpoint=10)

    args['lam'] = 50
    args['eta'] = 50

    loss_list, b_loss_list, tot_loss_list, pinn_loss_list = train(model=q,
                                                                  w=w,
                                                                  data=data,
                                                                  batchsize=batch_size,
                                                                  data_b=data_b,
                                                                  label_b=label_b,
                                                                  alpha_b=10,
                                                                  lr=1e-4,
                                                                  num_tsteps=10,
                                                                  num_epoches=Nsteps,
                                                                  args=args,
                                                                  device=device,
                                                                  dU=dU,
                                                                  checkpoint=10)
    t = np.arange(len(loss_list))  # Time values


# Create a figure with 3 subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # Plot training loss
    axs[0].plot(t, loss_list, label='Training Loss', color='blue')
    axs[0].set_title('Training Loss vs Time')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid()

    # Plot batch loss
    axs[1].plot(t, b_loss_list, label='Batch Loss', color='orange')
    axs[1].set_title('Boundary Loss vs Time')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid()

    # Plot total loss
    axs[2].plot(t, tot_loss_list, label='Total Loss', color='green')
    axs[2].set_title('Total Loss vs Time')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t, pinn_loss_list, label='Pinn Loss', color='red')
    axs[3].set_title('Pinn Loss vs Time')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Loss')
    axs[3].legend()
    axs[3].grid()

    # Adjust layout
    plt.tight_layout()
    
