#######################################################################################
################                                                       ################
################ This code is for using multi-GPU in VESSL AI Cluster  ################
################                                                       ################
#######################################################################################


from Model import Autoencoder

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
def set_seed (the_seed = 24):
  torch.manual_seed(the_seed)
  torch.cuda.manual_seed(the_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(the_seed)
  random.seed(the_seed)
  

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

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter    
    )
    
    parser.add_argument(
        "--backend", type=str, help="Distributed backend (NCCL or gloo)", default="nccl"
    )
    parser.add_argument(
        "--num_epochs", type=int, help="Number of training epochs.", default=10
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=5,
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate.", default=1e-3
    )
    
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=24)
    
    parser.add_argument(
        "--model_dir", type=str, help="Directory for saving models and outputs.", default="/output"
    )

    parser.add_argument(
        "--input_dir", type=str, help="Directory for input files.", default="/input"
    )
    
    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model filename.",
        default="Trained_model.pth",
    )
    
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved checkpoint."
    )
    args = parser.parse_args()        

    # Set random seeds to initialize the model
    set_seed(the_seed=args.random_seed)
    
    # Initializes the distributed backend to synchronize nodes/GPUs
    torch.distributed.init_process_group(backend=args.backend)

    # kodu path'den bagimsiz hale getirelim.
    # Kodunuzun ana dizinini bulun
    main_path = os.path.dirname(os.path.abspath(__file__))

    # Veri dosyalarının bulunduğu dizini belirleyin
    dataset_path = os.path.join(main_path, "..", "..", args.input_dir)

    # Output dosyalarinin cikacagi path
    output_path = os.path.join(main_path, "..", "..", args.model_dir)
     
    # Model with DDP
    local_rank = int(os.environ['LOCAL_RANK'])
    AE_model = Autoencoder()
    device = torch.device("cuda:{}".format(local_rank))
    AE_model = AE_model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        AE_model, device_ids= [args.local_rank], output_device=local_rank)
        
    # To resume, load model from "cuda:0"
    if args.resume:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(output_path, map_location=map_location))    
    
    
    # Prepare Dataset and dataLoader
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
    train_data = torch.tensor(train_data)
    val_data = torch.tensor(val_data)
    
    # Using distributed sampler for training dataset
    train_sampler = DistributedSampler(dataset=train_data)    
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=minibatch, 
        sampler=train_sampler,
        shuffle=False,
        num_workers=8
    )

    # Skip sampler for test_dataset 
    val_loader = DataLoader(
        val_data, 
        batch_size=val_data.shape[0], 
        shuffle=False,
        num_workers=8)



    #   Root mean square error
    criterion = nn.MSELoss()
    def RMSELoss(recon_x,x):
        return torch.sqrt(criterion(recon_x,x))

    ## after this point, define them as key-value pairs by command, hyperparameters using args lib
    #lr_step_size = 500
    #lr_gamma = 0.2

    #total_params = sum(p.numel() for p in AE_model.parameters())
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    #scheduler = lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma)
    start = time.time()

    # save the model after training
    model_filepath = os.path.join(output_path, args.model_filename)
    
    loss_his = []
    durations = []
    for epoch in tqdm(range(args.num_epochs)):
        #monitor training loss and validation loss
        print(f"Local Rank: {local_rank} Epoch: {epoch}, training started.")
        loss = 0.0
        start = time.time()
        ddp_model.train()
        
        # Empty the outputs list at the beginning of each epoch
        outputs = []
        latent_set = []
        ###################
        # train the model #
        ###################
        for idx, (images) in enumerate(train_loader):
            images = images.to(device)
            recon_images, latent_v = ddp_model(images)
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
    
        # Save and evaluate model only in local_rank 0
        if local_rank == 0:
            final_output_val, latent_val, accuracy, loss_val = evaluate(
                model=ddp_model, device=device, test_dataloader=val_loader, criterion=criterion
            )
            torch.save(ddp_model.state_dict(), model_filepath)
            print("-" * 75)
            print(
                f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss:.2f}, Elapsed: {duration:.2f}s"
            )
            print("-" * 75)

            # Logging to vessl
            vessl.log(
                step=epoch,
                payload={"accuracy": accuracy, "loss": loss, "elapsed": duration},
            )

        if epoch % 10 == 0:# and epoch != 0:
      
            to_print = "Epoch[{}/{}] Time: {:.0f} Loss: {:.6f}".format(epoch+1, 
                              args.num_epochs, time.time()-start, loss.item())
            print(to_print)    

    print(f"Total training time: {sum(durations):.2f}s")
    print(f"Average training time : {sum(durations) / len(durations):.2f}s")
    
    # concatenate outputs list to get the final output tensor and latent set tensor
    final_output = torch.cat(outputs, dim=0)
    final_latent_set = torch.cat(latent_set, dim=0)

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

if __name__ == "__main__":
    main()
