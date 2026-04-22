import sys
import os
origin_directory = os.getcwd()
model_directory = os.path.join(origin_directory, 'ala2')
src_directory = os.path.join(origin_directory, 'src')
sys.path.append(src_directory)
sys.path.append(model_directory)
import torch
import torch.optim as optim
import numpy as np
import statsmodels.api as sm
from torch.utils.data import DataLoader, TensorDataset
from nn import FunctionModel, save_model, load_model
import copy
import matplotlib.pyplot as plt
from model_training import train_resample,pinn_loss,build_rightside, train_mass
from hist import hist_reweight
from utils import *
from fes import U_phipsi,U_phitheta,phi_contour_1,psi_contour_1,phi_contour_2,theta_contour_2
from pathlib import Path
import re
from scipy import stats
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman','Times','DejaVu Serif']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Figure size in inches (example for 2x6 grid)
nrows, ncols = 5, 5
subplot_w, subplot_h = 8, 3
fig = plt.figure(figsize=(ncols*subplot_w, nrows*subplot_h))
axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]

labels = list('abcdefghijklmnopqrstuvwxyz')
for ax, lab in zip(axes, labels):
    ax.text(-0.1, 1.02, f'({lab})', transform=ax.transAxes,
            ha='left', va='top', fontsize=10, fontweight='bold')




epsilon = 1e-10

gamma_o = 10040
kbt = 300 * 0.0083144621  # kBT in kcal/mol 
num_heavy_atoms = heavy_atom_indices.shape[0]
xdim = heavy_atom_indices.shape[0] * 3
vdim = heavy_atom_indices.shape[0] * 3




#data_label = 'long'
#data_label = 'long_1'
#data_label = 'long_2'
#data_label = 'long_3'
#data_label = 'biased'
#data_label = 'all'

#xdim_reduce = 45
#vdim_reduce = 45  

activ  = 'sigmoid'
device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

#mass info
topology_file = "ala2/simulation/topol.top"
mass = read_mass(topology_file)
mass=torch.tensor(mass)
heavy_atom_mass = mass[heavy_atom_indices]
mass = torch.repeat_interleave(mass,3).unsqueeze(0)
heavy_atom_mass = torch.repeat_interleave(heavy_atom_mass,3).unsqueeze(0)
num_heavy_atoms = heavy_atom_indices.shape[0]

# data loading





'''
if data_label =='highT':
    data = data_T
    dU = dU_T
elif data_label == 'long':
    data = data_long
    dU = dU_long
elif data_label == 'long_1':
    data = data_long_1
    dU = dU_long_1
elif data_label == 'long_2':
    data = data_long_2
    dU = dU_long_2
elif data_label == 'long_3':
    data = data_long_3
    dU = dU_long_3
elif data_label == 'biased':
    data = torch.cat((data_long_1,data_long_2,data_long_3),dim = 0)
    dU = torch.cat((dU_long_1,dU_long_2,dU_long_3),dim = 0)
elif data_label == 'all':
    data = torch.cat((data_T,data_long,data_long_1,data_long_2,data_long_3),dim = 0)
    dU = torch.cat((dU_T,dU_long,dU_long_1,dU_long_2,dU_long_3),dim = 0)
'''







# q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=psi_group_heavy, activation='sigmoid')
#q = NNd2_45(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid')
#q = NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid')
#q.load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
#model_path_o = f"ala2/model/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.pth"
#config_path_o = f"ala2/config/distilling_gamma0.20005/gamma0.20005_highT_subtrain_2.txt"
#q0 = load_model_phipsi(model_path_o,config_path_o)


#q0.to(device)




def draw_q(qs,axes,gammas,data_labels,data,step,num_v,mmm,nnn,vs,k):
    num_models = len(qs)
    deg_to_rad = np.pi / 180.0
    deg_to_rad = np.float32(deg_to_rad)
    phi_angles = np.arange(-180, 180+step, step).astype(np.float32)
    psi_angles = np.arange(-60, 60+step, step).astype(np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_angles, psi_angles)
    phi_mesh = torch.from_numpy(phi_mesh * deg_to_rad).to(device)
    psi_mesh = torch.from_numpy(psi_mesh * deg_to_rad).to(device)
    q_values = torch.zeros((num_models,phi_mesh.shape[0], phi_mesh.shape[1])).to(device)
    q_values_at_v = torch.zeros((num_models,num_v, phi_mesh.shape[0], phi_mesh.shape[1])).to(device)

    with torch.no_grad():
        for i in range(phi_mesh.shape[0]):
            for j in range(phi_mesh.shape[1]):
                phi = int(phi_angles[j])
                psi = int(psi_angles[i])
                if phi == 180 or psi == 180:
                    continue
                print(f'Generating data for phi={phi}°, theta={psi}°')
                xs = read_xvg(f"ala2/simulation/constrained/phi_{phi}_psi_{psi}/positions.xvg")
                xs = torch.from_numpy(xs.astype(np.float32)).to(device)
                xs = xs[:, heavy_dim_indices]
                for iii in range(mmm):
                    for jjj in range(nnn):
                        idx = iii*nnn+jjj
                        v = vs[idx:idx+1,:]
                        v = v.repeat(xs.shape[0],1)
                        data_phi_psi = torch.cat((xs,v),dim=1)
                        data_phi_psi = data_phi_psi.to(device)
                        #print(data_phi_psi.shape)
                        
                        with torch.no_grad():
                            for iiii in range(num_models):
                                q = qs[iiii]
                                q_values_at_v[iiii,idx,i,j] = torch.mean(q(data_phi_psi)).detach()
                        del data_phi_psi
                        torch.cuda.empty_cache()

                
                repeated_xs = xs.repeat(k, 1)  # Shape: (k*m, n)

                # Step 2: Sample from normal distribution for the last n dimensions
                # Each column uses a different sigma
                sigmas = kbt/torch.sqrt(heavy_atom_mass).to(device)  # Shape: (1, n)
                #print(xs.shape,sigmas.shape)
                noise = torch.randn(k * xs.shape[0], xs.shape[1],device = device) * sigmas  # Shape: (k*m, n)

                # Step 3: Concatenate repeated_xs and noise along the second dimension
                data_phi_psi = torch.cat([repeated_xs, noise], dim=1)  # Shape: (k*m, 2n)
                data_phi_psi = data_phi_psi.to(device)
                
                for iiii in range(num_models):
                    q = qs[iiii]
                    q_values[iiii,i,j] = torch.mean(q(data_phi_psi)).detach()
                del data_phi_psi
                torch.cuda.empty_cache()
                #print(torch.std(q(data_phi_psi)))

                #print(phi, q.descriptor(data_phi_psi)[:,0].mean().item(),q.descriptor(data_phi_psi)[:,1].mean().item())
                #print(psi, q.descriptor(data_phi_psi)[:,2].mean().item(),q.descriptor(data_phi_psi)[:,3].mean().item())
                #print(i,j,q_value)



    phi_mesh, psi_mesh = np.meshgrid(phi_angles*deg_to_rad, psi_angles*deg_to_rad)
    q_values = q_values.detach().cpu().numpy()

    for iiii in range(num_models):
        q_values_i = q_values[iiii,:,:]
        ax = axes[iiii*ncols]
        cf = ax.contourf(phi_mesh,psi_mesh, q_values_i, cmap='turbo',levels=8)
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi/3, np.pi/3])
        fig.colorbar(cf,ax=ax)
        
        contour_lines = ax.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, colors='white')
        #contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, cmap="viridis")  # levels 控制等高线的数量
        plt.clabel(contour_lines, inline=True, fontsize=8)
        ax.set_xlabel('$\phi$ ')
        ax.set_ylabel('$\\theta$ ')

    for iiii in range(num_models):
        im = None
        for iii in range(mmm):
            for jjj in range(nnn):
                idx = iii*nnn+jjj
                ax = axes[iiii*ncols + idx +1]
                q_values_at_v_i = q_values_at_v[iiii,idx,:,:].detach().cpu().numpy()
                
                cf = ax.contourf(phi_mesh,psi_mesh, q_values_at_v_i, cmap='turbo',levels=8)
                fig.colorbar(cf,ax=ax)
                contour_lines = ax.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, colors='white')  # levels 控制等高线的数量
                ax.clabel(contour_lines, inline=True, fontsize=8)
                ax.set_xlim([-np.pi, np.pi])
                ax.set_ylim([-np.pi/3, np.pi/3])
                
                ax.set_xlabel('$\phi$')
                ax.set_ylabel('$\\theta$')
                #ax.set_title(f'Mean Committor Function at v {idx}')
                
                del q_values_at_v_i
        #plt.tight_layout()

    

        torch.cuda.empty_cache()


def draw_q_1(qs,axes,gammas,data_labels,data,step,num_v,mmm,nnn,vs,k):
    num_models = len(qs)
    deg_to_rad = np.pi / 180.0
    deg_to_rad = np.float32(deg_to_rad)
    phi_angles = np.arange(-180, 180+step, step).astype(np.float32)
    psi_angles = np.arange(-60, 60+step, step).astype(np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_angles, psi_angles)
    phi_mesh = torch.from_numpy(phi_mesh * deg_to_rad).to(device)
    psi_mesh = torch.from_numpy(psi_mesh * deg_to_rad).to(device)
    q_values = torch.zeros((num_models,phi_mesh.shape[0], phi_mesh.shape[1])).to(device)
    q_values_at_v = torch.zeros((num_models,num_v, phi_mesh.shape[0], phi_mesh.shape[1])).to(device)

    with torch.no_grad():
        for i in range(phi_mesh.shape[0]):
            for j in range(phi_mesh.shape[1]):
                phi = int(phi_angles[j])
                psi = int(psi_angles[i])
                if phi == 180 or psi == 180:
                    continue
                print(f'Generating data for phi={phi}°, theta={psi}°')
                xs = read_xvg(f"ala2/simulation/constrained/phi_{phi}_psi_{psi}/positions.xvg")
                xs = torch.from_numpy(xs.astype(np.float32)).to(device)
                xs = xs[:, heavy_dim_indices]
                for iii in range(mmm):
                    for jjj in range(nnn):
                        idx = iii*nnn+jjj
                        v = vs[idx:idx+1,:]
                        v = v.repeat(xs.shape[0],1)
                        data_phi_psi = torch.cat((xs,v),dim=1)
                        data_phi_psi = data_phi_psi.to(device)
                        #print(data_phi_psi.shape)
                        
                        with torch.no_grad():
                            for iiii in range(num_models):
                                q = qs[iiii]
                                q_values_at_v[iiii,idx,i,j] = torch.mean(q(data_phi_psi)).detach()
                        del data_phi_psi
                        torch.cuda.empty_cache()

                
                repeated_xs = xs.repeat(k, 1)  # Shape: (k*m, n)

                # Step 2: Sample from normal distribution for the last n dimensions
                # Each column uses a different sigma
                sigmas = kbt/torch.sqrt(heavy_atom_mass).to(device)  # Shape: (1, n)
                #print(xs.shape,sigmas.shape)
                noise = torch.randn(k * xs.shape[0], xs.shape[1],device = device) * sigmas  # Shape: (k*m, n)

                # Step 3: Concatenate repeated_xs and noise along the second dimension
                data_phi_psi = torch.cat([repeated_xs, noise], dim=1)  # Shape: (k*m, 2n)
                data_phi_psi = data_phi_psi.to(device)
                
                for iiii in range(num_models):
                    q = qs[iiii]
                    q_values[iiii,i,j] = torch.mean(q(data_phi_psi)).detach()
                del data_phi_psi
                torch.cuda.empty_cache()
                #print(torch.std(q(data_phi_psi)))

                #print(phi, q.descriptor(data_phi_psi)[:,0].mean().item(),q.descriptor(data_phi_psi)[:,1].mean().item())
                #print(psi, q.descriptor(data_phi_psi)[:,2].mean().item(),q.descriptor(data_phi_psi)[:,3].mean().item())
                #print(i,j,q_value)



    phi_mesh, psi_mesh = np.meshgrid(phi_angles*deg_to_rad, psi_angles*deg_to_rad)
    q_values = q_values.detach().cpu().numpy()


    for iiii in range(num_models):
        im = None
        for iii in range(mmm):
            for jjj in range(nnn):
                idx = iii*nnn+jjj
                ax = axes[iiii*ncols + idx]
                q_values_at_v_i = q_values_at_v[iiii,idx,:,:].detach().cpu().numpy()
                
                cf = ax.contourf(phi_mesh,psi_mesh, q_values_at_v_i, cmap='turbo',levels=8)
                fig.colorbar(cf,ax=ax)
                contour_lines = ax.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, colors='white')  # levels 控制等高线的数量
                ax.clabel(contour_lines, inline=True, fontsize=8)
                ax.set_xlim([-np.pi, np.pi])
                ax.set_ylim([-np.pi/3, np.pi/3])
                
                ax.set_xlabel('$\phi$')
                ax.set_ylabel('$\\theta$')
                #ax.set_title(f'Mean Committor Function at v {idx}')
                
                del q_values_at_v_i
        #plt.tight_layout()

    

        torch.cuda.empty_cache()

    '''
    for g in grid:
        phi_deg, psi_deg, _, _ = g
        dir_name = f"phi_{phi_deg}_psi_{psi_deg}"
        plot_colvar(
            f"{dir_name}/COLVAR_restrained",
            f"{dir_name}/COLVAR_restrained.png")
    '''

    ## need to fix for multiple models
    '''
    data = data.to(device)

    q_value = q(data)
    phipsi_value = phipsi(data[:,:xdim],num_heavy_atoms,phi_group_heavy,theta_group_heavy)
    plt.scatter(phipsi_value[:,0].detach().cpu().numpy(), phipsi_value[:,1].detach().cpu().numpy(), c=q_value.detach().cpu().numpy(), cmap='viridis', s=1)
    plt.colorbar(label='$q$')  # Add a colorbar to show the range of mean values
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.xlabel('$\phi$')
    plt.ylabel('$\\theta$')
    plt.savefig(figname_mq_data, dpi=300)
    plt.clf()

    q_value = q_value.detach().cpu().numpy()+1
    phipsi_value = phipsi_value.detach().cpu().numpy()
    args_hist = {}
    args_hist['xmin'] = phipsi_value[:,0].min()
    args_hist['xmax'] = phipsi_value[:,0].max()
    args_hist['ymin'] = phipsi_value[:,1].min()
    args_hist['ymax'] = phipsi_value[:,1].max()
    args_hist['xbins'] = 100
    args_hist['ybins'] = 100

    hist2d_m,xedges,yedges = hist2d_mean(phipsi_value[:,0], phipsi_value[:,1], q_value,args=args_hist)

    H, xedges, yedges = np.histogram2d(phipsi_value[:,0], phipsi_value[:,1], bins=[args_hist['xbins'], args_hist['ybins']], range=[[args_hist['xmin'], args_hist['xmax']], [args_hist['ymin'], args_hist['ymax']]])


    plt.figure(figsize=(6,5))
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.pcolormesh(xedges, yedges, hist2d_m, shading='auto', cmap='turbo')
    plt.colorbar(label='Mean Value')  # Add a colorbar to show the range of mean values
    plt.xlabel('$\phi$')
    plt.ylabel('$ \\theta$')
    plt.title('committor function ')
    plt.savefig(figname_q_data, dpi=300)
    plt.clf()

    plt.figure(figsize=(6,5))
    contour_lines = plt.contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, cmap="viridis")  # levels 控制等高线的数量
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.pcolormesh(xedges, yedges, np.log(H.T+epsilon), shading='auto', cmap='turbo')
    plt.colorbar(label='Mean Value')  # Add a colorbar to show the range of mean values
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Histogram with Mean Values')
    plt.savefig(figname_data_logp, dpi=300)
    plt.clf()
    torch.cuda.empty_cache()
    '''


def rate(model, data, weight,kbt,gamma, device):
        # data and weight should be sampled from the equilibrium distribution
        data = data.to(device)
        weight = weight.to(device)
        data.requires_grad_(True)


        qqq = model(data)
        with torch.no_grad():
            gradients = torch.autograd.grad(outputs=qqq, inputs=data,
                                            grad_outputs=torch.ones_like(qqq),
                                            create_graph=False, retain_graph=False)[0]
        if weight.shape is not (data.shape[0], 1):
            weight = weight.unsqueeze(dim=1)
        grad_v = gradients[:, xdim:]
        temp = weight * grad_v**2
        print(grad_v.shape,weight.shape,temp.shape)
        print(torch.mean(grad_v),torch.max(temp),torch.max(torch.sum(temp,dim = 1)),torch.sum(temp))
        return gamma * kbt * torch.sum(weight * (grad_v**2))

def rate_q0(model, data, weight, kbt,gamma, device):
        # data and weight should be sampled from the equilibrium distribution
        data = data.to(device)
        weight = weight.to(device)
        data.requires_grad_(True)


        qqq = model(data)
        with torch.no_grad():
            gradients = torch.autograd.grad(outputs=qqq, inputs=data,
                                            grad_outputs=torch.ones_like(qqq),
                                            create_graph=False, retain_graph=False)[0]
        if weight.shape is not (data.shape[0], 1):
            weight = weight.unsqueeze(dim=1)
        grad_v = gradients

        return  kbt * torch.sum(weight * (grad_v**2))


def quick_diagnose(axes,gammas,data_labels,layers,
    step = 10,
    num_v = 25,
    mmm = 2,
    nnn = 2,
    k = 1000,
    use_distance=False):
    qs = []
    figname_mqs = []
    figname_q_datas = []
    figname_mq_datas = []
    figname_data_logps = []
    figname_q_vslices = []
    filename_vs = f"ala2/model/vs_{kbt}.txt"
    vs = np.loadtxt(filename_vs)
    vs = torch.from_numpy(vs.astype(np.float32)).to(device)
    for gamma, data_label in zip(gammas, data_labels):
        if not use_distance:
            qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid'))
        else:
            qs.append(NNd2_45(layer_sizes=layers,n_atoms=num_heavy_atoms, activation='sigmoid'))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth"))
        qs[-1].to(device)
        if not os.path.exists(f"ala2/fig/gamma{gamma}"):
            os.makedirs(f"ala2/fig/gamma{gamma}")



    data_all = []
    draw_q_1(qs,axes,gammas,data_labels,data_all,step,num_v,mmm,nnn,vs,k)
    torch.cuda.empty_cache()

#if False:
if __name__ == '__main__':
    #layers = [xdim_reduce+vdim_reduce,8,64,64,64,64,8,1]
    use_distance = False
    if use_distance:
        xdim_reduce = 45
        vdim_reduce = 45
    else:
        xdim_reduce = 4
        vdim_reduce = 4
    layers = [xdim_reduce+vdim_reduce,8,256,256,1]
    layers_0 = [xdim_reduce, 64, 64, 1]
    gamma_data_label_file = "ala2/gamma_data_label.txt"
    gammas, data_labels = get_gamma_data_label(gamma_data_label_file)

    quick_diagnose(axes,gammas,data_labels,layers,step = 10,use_distance=use_distance,mmm=5,nnn=5)
    plt.savefig('ala2/fig/plot_metad.pdf', bbox_inches='tight')   # vector output


def draw_shooting(num_points,path,ax1,ax2,point_dir = 'point',shooting_result='result',bins = 20):
    results = []
    for i in range(num_points):
        shooting_file = f'{path}/{point_dir}_{i}/{shooting_result}'
        p = Path(shooting_file)
        if not p.exists():
            print(f'File {shooting_file} does not exist. Skipping.')
            continue
        text = p.read_text(encoding="utf-8")

        # look for patterns like "Shooting record: 8" (case-insensitive)
        patterns = {
            "shooting_record": re.compile(r"shooting\s*record\s*:\s*(-?\d+)", re.I),
            "num_simulations": re.compile(r"(?:number\s+of\s+simulations|num(?:ber)?\s+of\s+simulations)\s*:\s*(-?\d+)", re.I)
        }
        out: Dict[str, Optional[int]] = {k: None for k in patterns}
        for key, pat in patterns.items():
            m = pat.search(text)
            if m:
                out[key] = int(m.group(1))
        results.append(out["shooting_record"]/out["num_simulations"])

    
    

    
    counts, bins = np.histogram(results, bins=bins)
    # n are densities; convert to percent per bin by multiplying density by 100
    percent = counts / counts.sum() 
    bin_centers = (bins[:-1] + bins[1:]) / 2
    #ax1.bar(bin_centers, percent, width=bins[1]-bins[0], edgecolor='black')
    # optionally annotate percentages

    data=np.array(results)

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)  # 样本标准差

    # 绘图直方图（密度归一化）
    counts, bins, patches = ax1.hist(data, bins=bins, density=True, alpha=0.6, color='C0', edgecolor='k', label='Data histogram')

    
    # 绘制拟合正态密度曲线
    x = np.linspace(bins[0], bins[-1], 400)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax1.plot(x, pdf, 'r-', lw=2)

    # 可选：拟合检验打印
    ks_stat, ks_p = stats.kstest((data - mu)/sigma, 'norm')
    sh_stat, sh_p = stats.shapiro(data) if len(data) <= 5000 else (None, None)

    ax1.set_xlabel('Committor')
    ax1.set_ylabel('Density')
    
    

    print("Estimated mu =", mu)
    print("Estimated sigma =", sigma)
    print("KS test (vs normal): stat={:.4f}, p={:.4f}".format(ks_stat, ks_p))
    if sh_stat is not None:
        print("Shapiro-Wilk: stat={:.4f}, p={:.4f}".format(sh_stat, sh_p))

    

   
    
    fig = sm.qqplot((data - mu)/sigma, line='45',ax = ax2)
    



if False:
#if __name__ == "__main__":
    path = "ala2/simulation/iso/gamma5_biased_1"
    num_points = 300


    draw_shooting(num_points,path,axes[2],axes[3],bins=10)


    path = "ala2/simulation/iso/gamma25_biased_1"
    num_points = 300

    draw_shooting(num_points,path,axes[6],axes[7],bins=10)

    gamma_data_label_file = "ala2/find_iso_gamma_data_label.txt"
    gammas, data_labels = get_gamma_data_label(gamma_data_label_file)
    qs = []
    for gamma, data_label in zip(gammas, data_labels):
        positions_filename = "positions.xvg"
        velocities_filename = "velocities.xvg"
        forces_filename = "forces.xvg"
        xdim_reduce = 4
        vdim_reduce = 4
        layers = [xdim_reduce+vdim_reduce,8,256,256,1]
        activ  = 'sigmoid'
        device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")
        qs.append(NNphipsi(layer_sizes=layers,n_atoms=num_heavy_atoms,phi_group=phi_group_heavy,psi_group=theta_group_heavy, activation='sigmoid'))
        qs[-1].load_state_dict(torch.load(f"ala2/model/gamma{gamma}_kbt{kbt}_{data_label}.pth",map_location=torch.device('cpu')))
        qs[-1].to(device)

    data_biased = torch.tensor([]).to(device)
    
    data_biased_xv = torch.tensor([]).to(device)
    
    itrs=10
    for i in range(1,itrs+1):

        C7eq_path_biased = f"ala2/simulation/o_{100+i}/long_C7eq_{-3.5}/"
        C7ax_path_biased = f"ala2/simulation/o_{100+i}/long_C7ax_{-3.5}/"
        s_C7eq_xv,s_C7eq_fs,s_C7eq_xv_heavy,s_C7eq_fs_heavy = load_data(C7eq_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
        s_C7ax_xv,s_C7ax_fs,s_C7ax_xv_heavy,s_C7ax_fs_heavy = load_data(C7ax_path_biased,positions_filename,velocities_filename,forces_filename,heavy_atom_indices)
        
        
        data_biased = torch.cat((data_biased,s_C7eq_xv_heavy.to(device),s_C7ax_xv_heavy.to(device)),dim=0)
        data_biased_xv = torch.cat((data_biased_xv,s_C7eq_xv.to(device),s_C7ax_xv.to(device)),dim=0)
    
    data_biased = torch.cat((data_biased,data_biased),dim = 0)
    data_biased_xv = torch.cat((data_biased_xv,data_biased_xv),dim = 0)
    


    for i in range(len(gammas)):
        q = qs[i]
        gamma = float(gammas[i])
        data_label = data_labels[i]
        with torch.no_grad():
            q_values = torch.zeros((data_biased.shape[0],1)).to(device)
            NN = 100
            for ii in range(NN):
                vdim_origin = data_biased_xv.shape[1]//2
                #print(vdim_origin)
                data_biased_xv[:,vdim_origin:] = torch.randn(size=(data_biased_xv.shape[0],vdim_origin),
                                                            dtype=torch.float32) * np.sqrt(kbt)/torch.sqrt(mass)
                data_biased[:,xdim:] = data_biased_xv[:,vdim_origin+heavy_dim_indices]
                q_values += q(data_biased)
            q_values /= NN
            iso = (q_values - 0.5)**2<1e-4
            iso = iso.cpu()
            print(q(data_biased[iso.squeeze(),:]))
            iso_xv = data_biased_xv[iso.squeeze(),:]
            dihedrals_phipsi = phipsi(data_biased[iso.squeeze(),:xdim], num_heavy_atoms, phi_group_heavy, psi_group_heavy)
            dihedrals_phitheta = phipsi(data_biased[iso.squeeze(),:xdim], num_heavy_atoms, phi_group_heavy, theta_group_heavy)
            
            mask_phi = torch.abs(dihedrals_phitheta[:,0])<0.5
            mask_theta = torch.abs(dihedrals_phitheta[:,1])<0.5
            mask = mask_phi&mask_theta
            iso_xv = iso_xv[mask,:]
            iso_xv = iso_xv.detach().cpu()
            dihedrals_phipsi = dihedrals_phipsi[mask,:]
            dihedrals_phitheta = dihedrals_phitheta[mask,:]
            dihedrals_phipsi = dihedrals_phipsi.to('cpu').detach().numpy()
            dihedrals_phitheta = dihedrals_phitheta.to('cpu').detach().numpy()
            axes[i*ncols].scatter(dihedrals_phipsi[:,0],dihedrals_phipsi[:,1])
            
            contour_lines = axes[i*ncols].contour(phi_contour_1, psi_contour_1, U_phipsi, levels=8, cmap="viridis")
            plt.clabel(contour_lines, inline=True, fontsize=8)
            axes[i*ncols].set_xlim([-np.pi, np.pi])
            axes[i*ncols].set_ylim([-np.pi, np.pi])
            axes[i*ncols].set_xlabel('$\phi$')
            axes[i*ncols].set_ylabel('$\psi$')
            
            axes[i*ncols+1].scatter(dihedrals_phitheta[:,0],dihedrals_phitheta[:,1])
            contour_lines = axes[i*ncols+1].contour(phi_contour_2, theta_contour_2, U_phitheta, levels=8, cmap="viridis")
            plt.clabel(contour_lines, inline=True, fontsize=8)
            axes[i*ncols+1].set_xlim([-np.pi, np.pi])
            axes[i*ncols+1].set_ylim([-np.pi, np.pi])
            axes[i*ncols+1].set_xlabel('$\phi$')
            axes[i*ncols+1].set_ylabel('$\\theta$')
            
           
    plt.savefig('ala2/fig/plot_1.pdf', bbox_inches='tight')   # vector output        

