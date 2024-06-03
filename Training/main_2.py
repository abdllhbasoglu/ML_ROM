# in this source code, the training data is not shuffled and latent vector set is exported in order to use in regression models.
# This code is adjusted to VESSL AI cluster environment 
# The hyperparameters are an entry from cluster experience creation page
#                  Hyperparameters:
#                                --num_epochs     (default: 10)
#                                --batch_size     (default: 5)
#                                --learning_rate  (default: 1e-3)
#                                --model_filename (default: Trainedmodel.pth)
#                                --loss_function_type  (default: WMSE) --> 3 options as a) MSE, b)RMSE and c) WMSE
#                                --loss_mode      (default: log)  --> 3 options as a) linear, b)exponential and c) log

# different models are trained by changing 1st line (from Model_X import Autoencoder)

#######################################################################################
################                                                       ################
################ This code is for using single GPU in VESSL AI Cluster ################
################                                                       ################
#######################################################################################


from Model_B14 import Autoencoder

import vessl
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
import time
import argparse
import os
import random
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Weighted-loss funtion
def loss_function(reconstructed, origin, device, alpha=1.0, beta=1.0, mode='log', h_dim=150, w_dim=498):
    
    # Move weight matrices to the correct device
    # Exponential Weight Matrix
    weight_exp = torch.exp(torch.linspace(0, 1, steps=h_dim)).unsqueeze(1).repeat(1, w_dim)
    weight_exp = ((weight_exp - weight_exp.min()) / (weight_exp.max() - weight_exp.min())).to(device)
    
    # Log Weight Matrix
    weight_log = torch.logspace(0, 1, steps=h_dim).unsqueeze(1).repeat(1, w_dim)
    weight_log = ((weight_log - weight_log.min()) / (weight_log.max() - weight_log.min())).to(device)
    
    # WMSE-exp for pressure (the first channel)
    pressure_loss = weight_exp * (reconstructed[:, 0, :, :] - origin[:, 0, :, :]) ** 2
    pressure_loss = alpha * pressure_loss.mean()

    if mode == 'MSE':
        # (a) MSE for u-velocity and v-velocity (the second and third channels)
        u_velocity_loss = (reconstructed[:, 1, :, :] - origin[:, 1, :, :]) ** 2
        v_velocity_loss = (reconstructed[:, 2, :, :] - origin[:, 2, :, :]) ** 2
        velocity_loss = beta * (u_velocity_loss.mean() + v_velocity_loss.mean())    

    elif mode == 'log':
        # (b) WMSE-log for u-velocity and v-velocity (the second and third channels)
        u_velocity_loss = weight_log * (reconstructed[:, 1, :, :] - origin[:, 1, :, :]) ** 2
        v_velocity_loss = weight_log *(reconstructed[:, 2, :, :] - origin[:, 2, :, :]) ** 2
        velocity_loss = beta * (u_velocity_loss.mean() + v_velocity_loss.mean())
    
    # Total loss
    total_loss = pressure_loss + velocity_loss
    
    return total_loss

# Mean square error (MSE)
criterion = nn.MSELoss()

# Root mean square error (RMSE)
def RMSELoss(recon_x,x):
    return torch.sqrt(criterion(recon_x,x))

# to control randomization, a seed value is assigned  to make code repeatable
my_seed = 24
def set_seed (the_seed = 24):
  torch.manual_seed(the_seed)
  torch.cuda.manual_seed(the_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(the_seed)
  random.seed(the_seed)
  
set_seed(the_seed=my_seed)

# to evaluate the performance of the model for validation(test) dataset !!!!! bu fonskiyon uzerine daha fazla KAFA YORULMALI, ciktilari, Hata hesaplamalarina yonelik
def evaluate(model, test_dataloader, loss_fn_name, loss_mode, device):
    model.eval()
    diff, total_N = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            images = data.to(device)
            recon_images, latent = model(images)
            
            # loss function is also decided by assigned hyperparameter
            if loss_fn_name == "MSE":
                loss = criterion(recon_images, images)
            elif loss_fn_name == "RMSE":
                loss = RMSELoss(recon_images, images)
            elif loss_fn_name == "WMSE":
                loss = loss_function(reconstructed=recon_images, origin= images, device= device, mode = loss_mode)
            
            total_N += images.size(0)
            diff += (torch.abs(recon_images - images)/images).sum().item()

    MAPE = (diff / total_N)*100
    return recon_images, latent, MAPE, loss


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter    
    )

    # argument for number of epochs
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs.", default=10
    )
    
    # argument for batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=5,
    )
    
    # argument for learning rate
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate.", default=1e-3
    )
    
    # argument for trained model filename
    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model filename.",
        default="Trained_model.pth",
    )
    
    # argument for loss function selection
    parser.add_argument(
        "--loss_function_type",
        type=str,
        help="loss function type (MSE, RMSE or WMSE)",
        default="WMSE",
    )  

    # argument for weighted loss function mode
    parser.add_argument(
        "--loss_mode",
        type=str,
        help="Weighted loss function mode",
        default="log",
    )    
    
    args = parser.parse_args()   

    print(args.loss_function_type) #!!!!!!!!!!!! daha sonra silinecek
    
    # kodu path'den bagimsiz hale getirelim.
    # Kodunuzun ana dizinini bulun
    main_path = os.path.dirname(os.path.abspath(__file__))

    # Veri dosyalarının bulunduğu dizini belirleyin
    dataset_path = os.path.join(main_path, "..", "..", "input")

    # Output dosyalarinin cikacagi path
    output_path = os.path.join(main_path, "..", "..", "output")
    
    # Veri dosyalarını yükleme
    data = np.load(os.path.join(dataset_path, "flowfield_hom.npy"))
    flow_cond = np.load(os.path.join(dataset_path, "flowcon_hom.npy"))

    minibatch = args.batch_size # how many samples per batch to load
    if minibatch == None:
        minibatch = data.shape[0]


    # normalizing the flow field
    flowfield_mean = np.mean(data, axis=0) # cell-based mean values
    flowfield_std = np.std(data, axis=0) # cell-based std values
    normalized_data = (data - flowfield_mean) / flowfield_std


    # splitting train and validation data (coordinates, AoA whole together shuffled)
    train_data, val_data, train_flowcon, val_flowcon = train_test_split(normalized_data, flow_cond, test_size=0.2)


    #   make numpy to tensor
    train_loader = DataLoader(torch.tensor(train_data), batch_size=minibatch, shuffle=False)
    val_loader = DataLoader(torch.tensor(val_data), batch_size=val_data.shape[0], shuffle=False)
    
    ## after this point, define them as key-value pairs by command, hyperparameters using args lib
    # number of epochs
    #lr_step_size = 500
    #lr_gamma = 0.2
    AE_model = Autoencoder().to(device)
    total_params = sum(p.numel() for p in AE_model.parameters())
    print(total_params)
    optimizer = optim.Adam(AE_model.parameters(), lr=args.learning_rate)
    #scheduler = lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma)
    start = time.time()

    loss_his = []
    durations = []
    for epoch in tqdm(range(args.num_epochs)):
        #monitor training loss and validation loss
        loss = 0.0
        start = time.time()
        AE_model.train()
        
        # Empty the outputs list at the beginning of each epoch
        outputs = []
        latent_set = []
        ###################
        # train the model #
        ###################
        for idx, (images) in enumerate(train_loader):
            images = images.to(device)
            recon_images, latent_v = AE_model(images)
            
            # loss function is also decided by assigned hyperparameter
            if args.loss_function_type == "MSE":
                loss = criterion(recon_images, images)
            elif args.loss_function_type == "RMSE":
                loss = RMSELoss(recon_images, images)
            elif args.loss_function_type == "WMSE":
                loss = loss_function(reconstructed=recon_images, origin= images, device= device, mode = args.loss_mode) 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs.append(recon_images)
            latent_set.append(latent_v)  # append the latent vector
        loss_his.append(loss.item())
        #scheduler.step()
        
        end = time.time()
        duration = end - start
        durations.append(duration)          

        if epoch % 10 == 0:# and epoch != 0:
      
            to_print = "Epoch[{}/{}] Time: {:.0f} Loss: {:.6f}".format(epoch+1, 
                              args.num_epochs, time.time()-start, loss.item())
            print(to_print)
            
            # Also track validation loss 
            AE_model.eval()
            with torch.no_grad():
                for data in val_loader:
                    val_images = data.to(device)
                    val_recon_images, latent = AE_model(val_images)
            
                    # loss function is also decided by assigned hyperparameter
                    if args.loss_function_type == "MSE":
                        loss_val = criterion(val_recon_images, val_images)
                    elif args.loss_function_type == "RMSE":
                        loss_val = RMSELoss(val_recon_images, val_images)
                    elif args.loss_function_type == "WMSE":
                        loss_val = loss_function(reconstructed=val_recon_images, origin= val_images, device= device, mode = args.loss_mode)          

            # Logging to vessl
            vessl.log(
                step=epoch,
                payload={"loss_train": loss, "loss_val": loss_val, "elapsed": duration},
            )

    print(f"Total training time: {sum(durations):.2f}s")
    print(f"Average training time : {sum(durations) / len(durations):.2f}s")
 
    # save the model after training
    model_filepath = os.path.join(output_path, args.model_filename)
    torch.save(AE_model.state_dict(), model_filepath)
    
    # concatenate outputs list to get the final output tensor and latent set tensor
    final_output = torch.cat(outputs, dim=0)
    final_latent_set = torch.cat(latent_set, dim=0)

    # Evaulation of the validation set
    final_output_val, latent_val, MAPE_val, loss_val = evaluate(model= AE_model, 
                                                                test_dataloader=val_loader, loss_fn_name= args.loss_function_type, loss_mode=args.loss_mode, device=device)
    
    print("Validation Loss: ", MAPE_val, " and ", loss_val, ". (MAPE and Loss function, respectively)")

    flowfield_std = torch.tensor(flowfield_std).to(device)
    flowfield_mean = torch.tensor(flowfield_mean).to(device)
    
    # changing normalized data back to orginal distribution for training dataset
    reconstructed_data = torch.zeros(len(train_data),3,150,498).to(device)
    for i in range(3):
        for k in range(len(train_data)):
            reconstructed_data[k,i,:,:] = final_output[k,i,:,:] * flowfield_std[i] + flowfield_mean[i]

    # Origininal data
    train_data = torch.tensor(train_data).to(device)
    original_data = torch.zeros(len(train_data),3,150,498).to(device)
    for i in range(3):
        for k in range(len(train_data)):
            original_data[k,i,:,:] = train_data[k,i,:,:] * flowfield_std[i] + flowfield_mean[i]
    
    # error data
    error_data = abs(original_data - reconstructed_data) / original_data

    # changing normalized data back to orginal distribution for validation dataset
    reconstructed_val = torch.zeros(len(val_data),3,150,498).to(device)
    for i in range(3):
        for k in range(len(val_data)):
            reconstructed_val[k,i,:,:] = final_output_val[k,i,:,:] * flowfield_std[i] + flowfield_mean[i]

    # Exporting files to designed output file  / bagimsiz hale getirmeli bunlari da 
    # torch to numpy for latent vector and exporting
    latent_set_np = final_latent_set.cpu().detach().numpy()
    latent_set_filename = os.path.join(output_path, "latent_set.npy")
    np.save(latent_set_filename, latent_set_np)
    
    # torch to numpy for reconstructed data and exporting
    reconstructed_data = reconstructed_data.cpu().detach().numpy()
    reconstructed_data_filename = os.path.join(output_path, "reconstructed_data.npy") 
    np.save(reconstructed_data_filename, reconstructed_data)

    # torch to numpy for reconstructed validation data and exporting
    reconstructed_val = reconstructed_val.cpu().detach().numpy()
    reconstructed_val_filename = os.path.join(output_path, "reconstructed_val.npy") 
    np.save(reconstructed_val_filename, reconstructed_val)
    
    # Exporting the error
    error_data = error_data.cpu().detach().numpy()
    error_data_filename = os.path.join(output_path, "error_data.npy")
    np.save(error_data_filename, error_data)
    
    #shuffled AoA exporting
    shuff_train_flowcon = train_flowcon
    AoA_filename = os.path.join(output_path, "shuffled_t_AoA.npy") 
    np.save(AoA_filename, shuff_train_flowcon)

    #shuffled AoA for Val exporting
    shuff_val_flowcon = val_flowcon
    AoA_filename = os.path.join(output_path, "shuffled_v_AoA.npy") 
    np.save(AoA_filename, shuff_val_flowcon)
    
if __name__ == "__main__":
    main()
