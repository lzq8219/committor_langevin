import matplotlib.pyplot as plt
import numpy as np
import torch

def training_loss_plot(losses, title='Training Loss'):
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