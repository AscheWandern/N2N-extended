from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNetimport torch
from dataset_utils import AugmentNoise, DataLoader_Validation
from utils import calculate_ssim, calculate_psnr

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--models_dir_path', type=str, default='./pretrained_model')
parser.add_argument('model_path', type=str, default='model_gauss25_b4e100r02.pth')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)


opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

# Validation dataset
Kodak_dir = os.path.join(opt.val_dirs, "Kodak")   ### Directorio del dataset Kodak
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")   ### Directorio del dataset BSD300
Set14_dir = os.path.join(opt.val_dirs, "Set14")   ### Directorio del dataset Set14

validation_loader = DataLoader_Validation(Kodak_dir, BSD300_dir, Set14_dir)
valid_dict = {
    "Kodak": validation_loader.kodak(),
    "BSD300": validation_loader.bsd300(),
    "Set14": validation_loader.set14()
}

valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}
validation_path = os.path.join(save_path, "validation")
os.makedirs(validation_path, exist_ok=True)


noise_adder = AugmentNoise(style=opt.noisetype)

# Create complete path for the model
load_path = os.path.join(opt.models_dir_path, opt.model_path)
# Creation of an empty model
network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
if opt.parallel:   ### En caso de incluir la opcion de paralelizacion, se activa
    network = torch.nn.DataParallel(network)
    
# Load the weights of the network and put it on evaluation mode
network.load_state_dict(torch.load(load_path))
network = network.cuda()

network.eval()

np.random.seed(101)

for valid_name, valid_images in valid_dict.items():   ### Para cada dataset de validacion
    psnr_result = []   ### Array para almacenar los psnr (Peak Signal Noise Ratio)
    ssim_result = []   ### Array para almacenar los ssim (Structural Similarity Index Measure)
    repeat_times = valid_repeat_times[valid_name]   ### Se obtiene cuantas veces va a procesarse cada dataset
    for i in range(repeat_times):   ### 
        for idx, im in enumerate(valid_images):   ### Para cada imagen del dataset actual
            origin255 = im.copy()   ### Se crea una copia de la imagen original
            origin255 = origin255.astype(np.uint8)   ### Se convierte a tipo entero sin signo
            im = np.array(im, dtype=np.float32) / 255.0   ### Se convierte a numpy array de tipo float 32
            noisy_im = noise_adder.add_valid_noise(im)   ### Añade ruido artificial a la imagen preparada
            if epoch == opt.n_snapshot:   ### Si debe hacerse punto de guardado en esta epoca
                noisy255 = noisy_im.copy()   ### Se hace una copia de la imagen ruidosa
                noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)   ### Se desnormaliza la imagen, limitando los valores de los pixeles entre 0 y 255  (np.clip trunca los valores menores y mayores que el minimo y maximo indicado)
            # padding to square
            H = noisy_im.shape[0]   ### Se obtiene el alto de la imagen
            W = noisy_im.shape[1]   ### Se obtiene el ancho de la imagen
            val_size = (max(H, W) + 31) // 32 * 32
            noisy_im = np.pad(
                noisy_im,
                [[0, val_size - H], [0, val_size - W], [0, 0]],
                'reflect')   ### 
            transformer = transforms.Compose([transforms.ToTensor()])   ### Prepara un transformador a tensor
            noisy_im = transformer(noisy_im)   ### Convierte la imagen ruidosa en un tensor
            noisy_im = torch.unsqueeze(noisy_im, 0)   ### Devuelve la imagen tensor dentro de otro tensor en la posicion 0
            noisy_im = noisy_im.cuda()   ### Traslada el tensor a la GPU para operar con ella
            with torch.no_grad():   ### Desactiva el calculo del gradiente para no retornar el error a la red
                prediction = network(noisy_im)   ### Pasa la imagen ruidosa por la red y almacena la salida
                prediction = prediction[:, :, :H, :W]   ### Extrae del resultado la imagen, eliminando las capas que no interesan
            prediction = prediction.permute(0, 2, 3, 1)   ###Intercambia las dimensiones de la imagen
            prediction = prediction.cpu().data.clamp(0, 1).numpy()   ### Pasa la imagen a CPU y trunca los datos entre 0 y 1
            prediction = prediction.squeeze()   ### Elimina las dimensiones de tamaño 1
            pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                255).astype(np.uint8)   ### Desnormaliza la imagen devolviendola a valores de pixel entre 0 y 255
            # calculate psnr
            cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                        pred255.astype(np.float32))   ### Calcula el psnr entre la imagen original y la obtenida
            psnr_result.append(cur_psnr)   ### Almacena el psnr obtenido
            cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                        pred255.astype(np.float32))   ### Calcula el ssim entre la imagen original y la obtenida
            ssim_result.append(cur_ssim)   ### Almacena el ssim obtenido

    psnr_result = np.array(psnr_result)   ### Convierte el array de los psnr a tipo numpy
    avg_psnr = np.mean(psnr_result)   ### Obtiene la media del psnr de todas las imagenes evaluadas
    avg_ssim = np.mean(ssim_result)   ### Obtiene la media del ssim de todas las imagenes evaluadas
    log_path = os.path.join(validation_path,
                            "A_log_{}.csv".format(valid_name))   ### Crea la ruta para guardar un fichero de registro
    with open(log_path, "a") as f:   ### Crea un flujo de escritura apuntando al final del fichero
        f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))   ### Escribe en el fichero la epoca y las medidas obtenidas
