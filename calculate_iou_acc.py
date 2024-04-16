
import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10'  # Sets timeout to 2 seconds

import shutil
import re
import torch
import torchvision.io as io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
import csv
from PIL import Image
import cv2
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_iou(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred)
    union = torch.logical_or(y_true, y_pred)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou

def reverse_zeros_ones_bitwise(arr):
    arr = arr ^ 1  # Using XOR (^) to swap 0s and 1s
    return arr

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
    y_pred = reverse_zeros_ones_bitwise(y_pred)

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


if __name__=="__main__":

    # epoch_list = [10, 50, 100, 150, 200, 250] #, 100, 150, 200, 250
    # epoch_list = [5] #
    # epoch_list = [1400, 1500, 1600, 1700, 1800, 1900, 2000] #
    epoch_list = [100]#, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
    # epoch_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] #
    method = "kmeans"
    phase = "train"
    # model = "trainA2trainB_1024p_no_aug_spatial_attention"
    # model = "trainA2trainB_1024p_no_aug_spatial_attention_1000"
    model = "trainA2trainB_1024p_aug_scen_1_newDS_conference"
# 
    for epoch in epoch_list:
        print(epoch)
        
        
        # Source directory where your files are located
        # source_directory = 'D:\\JupyterLab\\Program\\colorSeparation\\colorseparationdataset\\[1] Sharp Spot Color 銳利色塊圖檔\\'
        source_directory = 'C:\\Users\\Yulius\\Desktop\\ground_truth_scen_1_aug\\'
        # generated_directory = 'D:\\JupyterLab\Program\\colorSeparation\\mask_generated\\'

        generated_directory_source = f'D:\\JupyterLab\\Demo\\Kmeans_SEAPix\\results\\{model}\\{method}\\{phase}_{epoch}\\'
        # generated_directory_source = f'D:\\JupyterLab\\Demo\pix2pixHD\\results\\trainA2trainB_1024p_aug_new_DS\\test_{epoch}\\'
        generated_directory = generated_directory_source+"mask_generated\\"
        destination_directory = generated_directory_source+"result\\"

        # Destination directory where you want to copy the file

        # create_dir(generated_directory_source+"result")
        # destination_directory = 'D:\\dataset_modify\\'

        # generated_directory_source = f'D:\\JupyterLab\\Demo\pix2pixHD\\results\\trainA2trainB_1024p_no_aug_spatial_attention\\{method}\\'
        # create_dir(generated_directory_source+f"result_{phase}_original_A")
        # generated_directory = generated_directory_source+f"mask_{phase}_original_A\\"
        # destination_directory = generated_directory_source+f"result_{phase}_original_A\\"

        num = 0
        best_filename_ypred = ""

        list_iou_all = []
        list_pixel_acc_all = []
        # Iterate through the files in the source directory

        data_per_epoch = []
        data_detail = []
        file_csv_per_epoch = f"{destination_directory}result_epoch_{phase}_{epoch}.csv"
        file_csv_detail = f"{destination_directory}\\result_epoch_{phase}_{epoch}_detail.csv"
        

        for folder_name in os.listdir(generated_directory):
            num += 1
            print(folder_name)
            folder_path = os.path.join(generated_directory, folder_name)
            list_iou = []
            list_pixel_acc = []

            checked_filenames = set()

            filename_y_pred_in_directory = os.listdir(folder_path)
            pattern = r'^(mask_)(.*\.png)$'
            filtered_files_y_pred = [filename for filename in filename_y_pred_in_directory if re.match(pattern, filename)]
            
            checked_filenames_zero = set()
            data = []
            count = 0

            for filename_y_pred in filtered_files_y_pred:
                file_csv = f"{destination_directory}\\{folder_name}\\{folder_name}.csv"
                
                parts = folder_name.split('_')
                sub_folder_num_color = parts[0]
                sub_folder_name = parts[1]
                sub_number_aug_name = parts[2]

                create_dir(destination_directory + folder_name)

                file_path_pred = os.path.join(folder_path, filename_y_pred)  # Get the complete path of the file
                best_filename_ypred = filename_y_pred
                if filename_y_pred not in checked_filenames_zero:
                    best_iou = 0.4 #minimal iou to skip, you can adjust it
                else:
                    if count == 1:
                        best_iou = 0.3
                    elif count == 2:
                        best_iou = 0.2
                    elif count == 3:
                        best_iou = 0.1
                    elif count == 4:
                        best_iou = 0.05
                    else:
                        best_iou = 0
                
                best_pixel_acc = 0
                best_filename = ""
                files_in_directory = os.listdir(os.path.join(source_directory, f"{sub_folder_num_color}\\{sub_folder_name}\\{sub_number_aug_name}"))
                # pattern = r'^\d+\.tif$'
                # pattern = r'^\d+\.(tif|tiff)$'
                pattern = r'^(AL_)(.*\.png)$'
                filtered_files = [filename for filename in files_in_directory if re.match(pattern, filename)]

                for filename_y_true in filtered_files:
                    if filename_y_true not in checked_filenames:
                        
                        
                        # # Load images using PyTorch
                        # y_pred = io.read_image(os.path.join(generated_directory, file_path_pred)).to('cuda')
                        # y_true = io.read_image(os.path.join(source_directory, f"{sub_folder_num_color}\{sub_folder_name}\{filename_y_true}")).to('cuda')

                        # Load TIFF images using PIL and convert to PyTorch tensors
                        y_pred = cv2.imdecode(np.fromfile(os.path.join(generated_directory, file_path_pred), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                        y_true = cv2.imdecode(np.fromfile(os.path.join(source_directory, f"{sub_folder_num_color}\\{sub_folder_name}\\{sub_number_aug_name}\\{filename_y_true}"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

                        if y_true.ndim > 2:
                            y_true = np.mean(y_true, axis=2)
                        
                        # new_width = 1024
                        # new_height = 1024

                        # y_true = cv2.resize(y_true, (new_width, new_height), interpolation = cv2.INTER_NEAREST)

                        iou, pixel_accuracy, y_true, y_pred = calculate_acc_iou(y_true, y_pred)

                        if iou >= best_iou:
                            best_iou = iou
                            best_pixel_acc = pixel_accuracy
                            best_filename = filename_y_true
                                
                        # # Save images using PyTorch
                        # y_pred_cpu = y_pred.cpu()
                        # y_true_cpu = y_true.cpu()

                        io.write_png((y_pred.cpu() * 255).to(torch.uint8).unsqueeze(0), os.path.join(destination_directory, f"{folder_name}\\check_{filename_y_pred.split('.png')[0]}.png"))
                        io.write_png((y_true.cpu() * 255).to(torch.uint8).unsqueeze(0), os.path.join(destination_directory, f"{folder_name}\\y_true_{filename_y_true.split('.')[0]}.png"))
                
                if best_filename != "":
                    checked_filenames.add(best_filename)
                    list_iou.append(best_iou)
                    list_pixel_acc.append(best_pixel_acc)
                    print(f"y_pred: {filename_y_pred} and y_true: {best_filename}")
                    print(f"Intersection over Union (IoU): {best_iou:.4f}")
                    print(f"Pixel Accuracy: {best_pixel_acc:.4f}")
                    data.append([filename_y_pred, best_filename, best_iou.item(), best_pixel_acc])
                    data_detail.append([folder_name, filename_y_pred, best_filename, best_iou.item(), best_pixel_acc])

                    print("\n")
                else:
                    filtered_files_y_pred.append(filename_y_pred)
                    checked_filenames_zero.add(filename_y_pred)
                    count+=1

                with open(file_csv, mode='w', newline='') as file:
                    csv_writer = csv.writer(file)
                    # Write the data row by row
                    csv_writer.writerow(["y_pred", "y_true", "IoU", "Pixel Accuracy"])
                    for row in data:
                        csv_writer.writerow(row)

            list_iou_all.append(torch.mean(torch.tensor(list_iou)))
            list_pixel_acc_all.append(torch.mean(torch.tensor(list_pixel_acc)))
            print(f"Mean IoU Per Folder: {torch.mean(torch.tensor(list_iou))}")
            print(f"Mean Pixel Acc: {torch.mean(torch.tensor(list_pixel_acc))}")        
            print("\n")
        
        print(f"Total Folder: {num}")
        print(f"Mean IoU All: {torch.mean(torch.tensor(list_iou_all))}")
        print(f"Mean Pixel Acc All: {torch.mean(torch.tensor(list_pixel_acc_all))}")

        data_per_epoch.append([epoch, num, torch.mean(torch.tensor(list_iou_all)).item(), torch.mean(torch.tensor(list_pixel_acc_all)).item()])

        with open(file_csv_per_epoch, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write the data row by row
            csv_writer.writerow(["epoch", "total folder", "mean iou all", "mean pixell acc all"])
            for row in data_per_epoch:
                csv_writer.writerow(row)

        with open(file_csv_detail, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write the data row by row
            csv_writer.writerow(["folder_name","y_pred", "y_true", "IoU", "Pixel Accuracy"])
            for row in data_detail:
                csv_writer.writerow(row)
        
