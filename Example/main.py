

# in this source code, the training data is not shuffled and latent vector set is exported in order to use in regression models.
# additionaly, all data is used as training data, we dont have any validation data so validation data related codes

from Model import Autoencoder
from utils import Reconstructed_writer # veriyi shuffle ediyorum lakin kaydederken bu shuffle olayini hesaba katmiyorum. o yuzden shuffle edilmis AoA'yi da yazdirsam iyi olur.

import vessl
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import time
import argparse
import os
import random
torch.set_default_dtype(torch.float64)

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
def evaluate(model, test_dataloader, criterion):
    model.eval()
    diff, total_N = 0, 0
    with torch.no_grad():
        for data in test_dataloader:
            images = data
            outputs, latent = model(images)
            loss = criterion(outputs, images)
            total_N += images.size(0)
            diff += (torch.abs(outputs - images)/images).sum().item()

    MAPE = (diff / total_N)*100
    return outputs, latent, MAPE, loss


def main():

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

    minibatch = 5 # how many samples per batch to load
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



    #   Root mean square error
    criterion = nn.MSELoss()
    def RMSELoss(recon_x,x):
        return torch.sqrt(criterion(recon_x,x))

    ## after this point, define them as key-value pairs by command, hyperparameters using args lib
    # number of epochs
    n_epochs = 300
    lr_step_size = 500
    lr_gamma = 0.2
    AE_model = Autoencoder()
    total_params = sum(p.numel() for p in AE_model.parameters())
    optimizer = optim.Adam(AE_model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma)
    start = time.time()

    loss_his = []
    durations = []
    for epoch in tqdm(range(n_epochs)):
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
            images = images
            recon_images, latent_v = AE_model(images)
            loss = criterion(recon_images, images)
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
                              n_epochs, time.time()-start, loss.item())
            print(to_print)    

    print(f"Total training time: {sum(durations):.2f}s")
    print(f"Average training time : {sum(durations) / len(durations):.2f}s")
 
    # save the model after training
    model_filepath = os.path.join(output_path, "Trained_model.pth")
    torch.save(AE_model.state_dict(), model_filepath)
    
    # concatenate outputs list to get the final output tensor and latent set tensor
    final_output = torch.cat(outputs, dim=0)
    final_latent_set = torch.cat(latent_set, dim=0)

    # Evaulation of the validation set
    final_output_val, latent_val, MAPE_val, loss_val = evaluate(model= AE_model, 
                                                                test_dataloader=val_loader, criterion=criterion)

    # changing normalized data back to orginal distribution for training dataset
    reconstructed_data = torch.zeros(len(train_data),3,150,498)
    for i in range(3):
        for k in range(len(train_data)):
            reconstructed_data[k,i,:,:] = final_output[k,i,:,:]* torch.tensor(flowfield_std)[i] + torch.tensor(flowfield_mean)[i]

    # Origininal data
    original_data = torch.zeros(len(train_data),3,150,498)
    for i in range(3):
        for k in range(len(train_data)):
            original_data[k,i,:,:] = torch.tensor(train_data[k,i,:,:])* torch.tensor(flowfield_std)[i] + torch.tensor(flowfield_mean)[i]
    
    # error data
    error_data = abs(original_data - reconstructed_data) / original_data

    # changing normalized data back to orginal distribution for validation dataset
    reconstructed_val = torch.zeros(len(val_data),3,150,498)
    for i in range(3):
        for k in range(len(val_data)):
            reconstructed_val[k,i,:,:] = final_output_val[k,i,:,:]* torch.tensor(flowfield_std)[i] + torch.tensor(flowfield_mean)[i]

    # Exporting files to designed output file  / bagimsiz hale getirmeli bunlari da 
    # torch to numpy for latent vector and exporting
    latent_set_np = final_latent_set.cpu().detach().numpy()
    latent_set_filename = os.path.join(output_path, "latent_set.npy")
    np.save(latent_set_filename, latent_set_np)
    
    # torch to numpy for reconstructed data and exporting
    reconstructed_data = reconstructed_data.cpu().detach().numpy()
    reconstructed_data_filename = os.path.join(output_path, "reconstructed_data.npy") 
    np.save(reconstructed_data_filename, reconstructed_data)

    # Exporting the error
    error_data = error_data.cpu().detach().numpy()
    error_data_filename = os.path.join(output_path, "error_data.npy")
    np.save(error_data_filename, error_data)
    
    #shuffled AoA exporting
    shuff_train_flowcon = train_flowcon
    AoA_filename = os.path.join(output_path, "shuffled_t_AoA.npy") 
    np.save(AoA_filename, shuff_train_flowcon)

if __name__ == "__main__":
    main()
