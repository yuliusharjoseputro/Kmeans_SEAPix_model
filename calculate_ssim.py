
from math import exp
import torch
import torchvision.io as io
import torch.nn.functional as F
from torch.autograd import Variable
import os
import re
import lpips
import glob
from sklearn.metrics import confusion_matrix
# import numpy as np


# Assuming other necessary imports are here

def ssim(image1, image2, K, window_size, L):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5      # default
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).to('cuda')
    _1D_window = (gauss/gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous()).to('cuda')
    
    C1 = K[0]**2
    C2 = K[1]**2
    
    mu1 = F.conv2d(image1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(image1*image1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(image2*image2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(image1*image2, window, padding = window_size//2, groups = channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def PSNR(real_images_tensor, generated_images_tensor):
    # Assuming real_images and generated_images are PyTorch tensors on GPU

    mse = ((real_images_tensor - generated_images_tensor) ** 2).mean()
    
    if mse == 0:  # MSE is zero means no noise is present in the signal
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def LPIPS(real_images_tensor, generated_images_tensor):

    lpips_model = lpips.LPIPS(net='alex').to('cuda')

    distance = lpips_model(real_images_tensor, generated_images_tensor)

    return distance

def mse(real_images_tensor, generated_images_tensor):
    
	# err = torch.mean((real_images_tensor - generated_images_tensor) ** 2)

    real_images_tensor = real_images_tensor.float()
    generated_images_tensor = generated_images_tensor.float()

	# Calculate squared difference
    squared_diff = (real_images_tensor - generated_images_tensor) ** 2

    # Manually compute the mean
    err = torch.sum(squared_diff) / (real_images_tensor.shape[2] * real_images_tensor.shape[3])
    
    return err

def calculate_iou(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred)
    union = torch.logical_or(y_true, y_pred)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou

def calculate_acc_iou(y_true, y_pred):
    # new_width = 1024
    # new_height = 1024

    # Convert the numpy arrays to PyTorch tensors and move them to GPU
    y_true = torch.tensor(y_true).to('cuda')
    y_pred = torch.tensor(y_pred).to('cuda')

    # # Resize using PyTorch
    # y_true = TF.resize(y_true, [new_height, new_width])
    # y_pred = TF.resize(y_pred, [new_height, new_width])

    y_true = (y_true / 255).int()
    y_pred = (y_pred / 255).int()

    # y_true = reverse_zeros_ones_bitwise(y_true)
    # y_pred = reverse_zeros_ones_bitwise(y_pred)

    # Flatten the tensors for confusion matrix calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Convert back to numpy for sklearn (confusion matrix calculation)
    y_true_flat_np = y_true_flat.cpu().numpy()
    y_pred_flat_np = y_pred_flat.cpu().numpy()

    # Calculate confusion matrix and IoU
    tn, fp, fn, tp = confusion_matrix(y_true_flat_np, y_pred_flat_np).ravel()

    # Calculate pixel accuracy
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn)

    #calculate IoU
    iou = calculate_iou(y_true, y_pred)

    return iou, pixel_accuracy, y_true, y_pred

def load_image_pytorch(image_path):
    image = io.read_image(image_path)
    
    return image



if __name__=="__main__":
    
    # epoch_list = [10, 50, 100, 150, 200, 250]
    epoch_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    method = "kmeans"
    phase = "train"
    framework = "PF1"
    # model = "trainA2trainB_1024p_no_aug_spatial_attention"
    if framework=="PF1":
        model = "trainA2trainB_1024p_aug_scen_1_newDS_conference"
    elif framework=="PF2":
        model = "trainA2trainB_1024p_no_aug_scen_2_newDS_Combined_Spatial_SE_Attention"

    for epoch in epoch_list:
        print(epoch)
        
        if framework=="PF1":
            # source_directory = 'C:\\Users\Yulius\\Downloads\\Compressed\\test\\test_latest\\images\\' #syntethize
            synthesized_directory = f'D:\\JupyterLab\\Demo\\\Kmeans_SEAPix\\results\\{model}\\{method}\\{phase}_{epoch}\\images\\' #syntethize
            # synthesized_directory = 'C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_no_aug\\test_A\\'
            GT_directory = f'C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_aug_scenario1_newDS_conference\\{phase}_B\\'

        elif framework=="PF2":
            # source_directory = 'C:\\Users\Yulius\\Downloads\\Compressed\\test\\test_latest\\images\\' #syntethize
            synthesized_directory = f'D:\\JupyterLab\\Demo\\pix2pixHD\\results\\{model}\\{method}\\{phase}_{epoch}\\images\\' #syntethize
            # synthesized_directory = 'C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_no_aug\\test_A\\'
            GT_directory = f'C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_no_aug_scenario2_newDS\\{phase}_B\\'
        
        # #train
        # synthesized_directory = f'D:\\JupyterLab\\Demo\\pix2pixHD\\results\\trainA2trainB_1024p_no_aug\\train_{epoch}\\images\\' #syntethize
        # GT_directory = 'C:\\Baseline_Dataset\\dataset_fix\\new_data_80_20_sharp_color_fold_1024_no_aug\\train_B\\'

        # GT_directory = 'D:\\JupyterLab\\Demo\\pix2pixHD\\datasets\\colorseparation\\test_B\\'

        # checkfiles()

        # default constants
        K = [0.01, 0.03]
        L = 255; 
        window_size = 11

        os.chdir(synthesized_directory)
        synthetize_list_filename=[]
        groundtruth_list_filename=[]

        synthetize_list=[]
        groundtruth_list=[]

        # List filenames containing 'synthesized_image'
        file_list = glob.glob("*synthesized_image*")
        # file_list = glob.glob("*input_label*")
        # file_list = glob.glob("*-*")
        # print("Filenames containing 'synthesized_image':")
        for file_name in file_list:
            # print(file_name)
            I1 = load_image_pytorch(synthesized_directory + file_name)
            I1 = I1.float() / 255.0  # Normalize to [0, 1]
            I1 = I1.unsqueeze(0)  # Add batch dimension
            I1 = I1.to('cuda')
            # I1 = torch.from_numpy(np.rollaxis(I1, 2)).float().unsqueeze(0)/255.0
            I1 = Variable(I1, requires_grad = True)

            synthetize_list.append(I1)
            synthetize_list_filename.append(file_name)
        
            # Split the filename by underscores
            parts = file_name.split('_')

            # Replace "A" with "AL" in the first part
            parts[0] = parts[0].replace("A", "AL")

            # Extract the desired part
            
            extension = ".png"

            if framework=="PF1":
                ground_truth_filename = '_'.join(parts[:4])
            elif framework=="PF2":
                ground_truth_filename = parts[0] + "_" + parts[1] + "_" + parts[2] + "_" + parts[3][1:] + "_" + parts[4]

            groundtruth_list_filename.append(ground_truth_filename+extension)
            # groundtruth_list_filename.append(ground_truth_filename)

        for second_file_name in groundtruth_list_filename:
            I2 = load_image_pytorch(GT_directory + second_file_name)
            I2 = I2.float() / 255.0  # Normalize to [0, 1]
            I2 = I2.unsqueeze(0)  # Add batch dimension
            I2 = I2.to('cuda')
            # I2 = torch.from_numpy(np.rollaxis(I2, 2)).float().unsqueeze(0)/255.0
            I2 = Variable(I2, requires_grad = True)

            groundtruth_list.append(I2)

        synthetize_list_filename.sort()
        groundtruth_list_filename.sort()

        ssim_list=[]
        psnr_list=[]
        fid_list=[]
        lpips_list=[]
        mse_list=[]

        if framework=="PF2":
            iou_list=[]
            pixel_accuracy_list=[]

        with open(synthesized_directory+f'..\\log_evaluation_{phase}_{epoch}.txt', 'w') as f:
            for i in range(len(synthetize_list)):

                real_images = load_image_pytorch(GT_directory + groundtruth_list_filename[i])
                generated_images = load_image_pytorch(synthesized_directory + synthetize_list_filename[i])

                ssim_value = ssim(synthetize_list[i], groundtruth_list[i], K, window_size, L)
                ssim_list.append(ssim_value.item())

                real_images_lpips = real_images.float() / 255.0  # Normalize to [0, 1]
                real_images_lpips = real_images_lpips.unsqueeze(0)  # Add batch dimension
                real_images_lpips = real_images_lpips.to('cuda')

                generated_images_lpips = generated_images.float() / 255.0  # Normalize to [0, 1]
                generated_images_lpips = generated_images_lpips.unsqueeze(0)  # Add batch dimension
                generated_images_lpips = generated_images_lpips.to('cuda')

                lpips_value = LPIPS(real_images_lpips, generated_images_lpips)
                lpips_list.append(lpips_value.item())

                if framework=="PF2":
                    iou, pixel_accuracy, y_true, y_pred = calculate_acc_iou(real_images, generated_images)
                    iou_list.append(iou.item())
                    pixel_accuracy_list.append(pixel_accuracy.item())

                #calculate PSNR
                psnr_value = PSNR(real_images_lpips, generated_images_lpips)
                psnr_list.append(psnr_value)

                real_images = real_images.float()
                real_images = real_images.unsqueeze(0)  # Add batch dimension
                real_images = real_images.to('cuda')

                generated_images = generated_images.float()
                generated_images = generated_images.unsqueeze(0)  # Add batch dimension
                generated_images = generated_images.to('cuda')
                
                mse_value = mse(real_images, generated_images)
                mse_list.append(mse_value)
                
                if framework=="PF1":
                    textPrint = synthetize_list_filename[i], " - ", groundtruth_list_filename[i], " - ",ssim_value.item(), " - ", psnr_value, "dB", " - ", lpips_value.item(), " - ", mse_value
                elif framework=="PF2":
                    textPrint = synthetize_list_filename[i], " - ", groundtruth_list_filename[i], " - ",ssim_value.item(), " - ", psnr_value, "dB", " - ", lpips_value.item(), " - ", mse_value, " - ", iou, " - ", pixel_accuracy

                print(str(textPrint))

                f.write(str(textPrint))
                f.write('\n')
            
            ssim_list_tensor = torch.tensor(ssim_list).to('cuda')
            psnr_list_tensor = torch.tensor(psnr_list).to('cuda')
            lpips_list_tensor = torch.tensor(lpips_list).to('cuda')
            mse_list_tensor = torch.tensor(mse_list).to('cuda')

            if framework=="PF2":
                iou_list_tensor = torch.tensor(iou_list).to('cuda')
                pixel_accuracy_list_tensor = torch.tensor(pixel_accuracy_list).to('cuda')
        
            f.write('\n')
            print("Average of SSIM: ",torch.mean(ssim_list_tensor).item())
            print("Average of PSNR: ",torch.mean(psnr_list_tensor.float()).item())
            print("Average of LPIPS: ",torch.mean(lpips_list_tensor).item())
            print("Average of MSE: ",torch.mean(mse_list_tensor).item())

            if framework=="PF2":
                print("Average of IoU: ",torch.mean(iou_list_tensor).item())
                print("Average of Pixel Acc: ",torch.mean(pixel_accuracy_list_tensor).item())
            
            f.write("Average of SSIM: ") 
            f.write(str(torch.mean(ssim_list_tensor).item()))
            f.write('\n')
            f.write("Average of PSNR: ")
            f.write(str(torch.mean(psnr_list_tensor.float()).item()))
            f.write('\n')
            f.write("Average of LPIPS: ")
            f.write(str(torch.mean(lpips_list_tensor).item()))
            f.write('\n')
            f.write("Average of MSE: ")
            f.write(str(torch.mean(mse_list_tensor).item()))

            if framework=="PF2":
                f.write("Average of IoU: ")
                f.write(str(torch.mean(iou_list_tensor).item()))
                f.write('\n')
                f.write("Average of Pixel Acc: ")
                f.write(str(torch.mean(pixel_accuracy_list_tensor).item()))