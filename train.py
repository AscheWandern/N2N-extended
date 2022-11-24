from __future__ import division
import os
import time
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader

from models.unet import UNet
from models.esrt import ESRT
from dataset_utils import AugmentNoise, DataLoader_Imagenet_val, DataLoader_Validation, crop_image
from noise_metrics import calculate_ssim, calculate_psnr
from generator_subimages import generate_mask_pair, generate_subimages
from utils import checkpoint
import settings

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument("--crop_size", type=int, default=None)
parser.add_argument("--torch_seed", type=int, default=3407)
parser.add_argument('--arch', type=str, required=True, \
            choices=["unet", "esrt"], \
            help='dataset (options: unet, esrt)')

opt, _ = parser.parse_known_args()  ### Recopilar parametros de ejecucion
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices   ### Selección de dispositivo gpu para la ejecucion
settings.init()


torch.manual_seed(opt.torch_seed)
# Training Set
TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)   ### Preparacion de la informacion para la carga del dataset
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=8,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)   ### Creacion del cargador del dataset de entrenamiento

# Validation Set
Kodak_dir = os.path.join(opt.val_dirs, "Kodak")   ### Directorio del dataset Kodak
BSD300_dir = os.path.join(opt.val_dirs, "BSD300")   ### Directorio del dataset BSD300
Set14_dir = os.path.join(opt.val_dirs, "Set14")   ### Directorio del dataset Set14

validation_loader = DataLoader_Validation(Kodak_dir, BSD300_dir, Set14_dir)
valid_dict = {
    "Kodak": validation_loader.kodak(),
    "BSD300": validation_loader.bsd300(),
    "Set14": validation_loader.set14()
}

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)   ### Creacion de objeto que servira para la inclusion de ruido artificial

# Network
if opt.arch == "unet":
    network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)   ### Creación de la red neuronal
elif opt.arch == "esrt":
    network = ESRT(in_nc=opt.n_channel, n_feature=opt.n_feature)
if opt.parallel:   ### En caso de incluir la opcion de paralelizacion, se activa
    network = torch.nn.DataParallel(network)
network = network.cuda()   ### Se pasa la red a la GPU para aligerar la computacion

# about training scheme
num_epoch = opt.n_epoch   ### Se establece el numero de epocas que tendra el entrenamiento
ratio = num_epoch / 100   ### Indice de hitos donde se haran cambios
optimizer = optim.Adam(network.parameters(), lr=opt.lr)   ### Se establece el optimizador (el que actualiza los parametros) por los indices indicados
scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                     ],
                                     gamma=opt.gamma)   ### Disminuye el valor del lr en las epocas indicadas en milestone
print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

checkpoint(network, 0, "model", opt.save_model_path, opt.log_name)   ### Creacion de un punto de guardado del modelo antes de entrenar
print('init finish')

### Comienzo del algoritmo de entrenamiento
for epoch in range(1, opt.n_epoch + 1):   
    cnt = 0   ### Se resetea el contador a 0 para cada epoca

    for param_group in optimizer.param_groups:   ### Extrae los diferentes lr que pueda haber y los muestra
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    network.train()   ### Activa la opcion de entrenar la red para que modifique sus parametros cuando las imagenes pasen por ella
    for iteration, clean in enumerate(TrainingLoader):   ### En cada epoca se cargan las imagenes una a una para introducirla a la red
        st = time.time()   ### Almacena el tiempo en el que comienza la iteracion
        clean = clean / 255.0   ### Normaliza la imagen (el valor de un pixel va de 0 a 255)
        clean = clean.cuda()   ### Mueve la imagen a la GPU 
        noisy = noise_adder.add_train_noise(clean)   ### Añade ruido a la imagen

        optimizer.zero_grad()   ### Limpia el optimizador para resetear el error acumulado

        mask1, mask2 = generate_mask_pair(noisy)   ### Genera las mascaras con las que se obtendran las subimagenes
        noisy_sub1 = generate_subimages(noisy, mask1)   ### Genera la primera subimagen ruidosa
        noisy_sub2 = generate_subimages(noisy, mask2)   ### Genera la segunda subimagen ruidosa
        with torch.no_grad():   ### Desactiva el calculo del gradiente para no retornar el error a la red
            noisy_denoised = network(noisy)   ### Obtiene una version limpia de la imagen completa
        ### Genera un par de subimagenes vecinas a partir de la imagen limpia que produce la red
        noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

        noisy_output = network(noisy_sub1)   ### La primera subimagen ruidosa se pasa por la red para obtener una salida limpia
        noisy_target = noisy_sub2   ### La segunda subimagen ruidosa se convierte en el resultado esperado
        Lambda = epoch / opt.n_epoch * opt.increase_ratio   ### Se calcula el parametro lambda
        diff = noisy_output - noisy_target   ### Se calcula la diferencia entre salida de la red y salida esperada
        exp_diff = noisy_sub1_denoised - noisy_sub2_denoised   ### Se calcula la diferencia entre las subimagenes de la imagen limpiada por la red

        loss1 = torch.mean(diff**2)   ### El primer indice de perdida es calculado como la media de la distancia de las subimagenes originales al cuadrado
        loss2 = Lambda * torch.mean((diff - exp_diff)**2)   ### El segundo indice de perdida se calcula calculando la media de la diferencia entre ambos errores al cuadrado multiplicado por lambda como regulador
        """Para mas informacion acudir al articulo"""
        
        loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2   ### Calculo de error total

        loss_all.backward()   ### Se propaga el error por la red acumulando el gradiente
        optimizer.step()   ### El optimizador actualiza los parametros en funcion del gradiente
        print(
            '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
            .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                    np.mean(loss2.item()), np.mean(loss_all.item()),
                    time.time() - st))

    scheduler.step()   ### Se indica al planificador que se ha realizado una etapa

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:   ### Si esta epoca esta marcada para realizar un guardado o es la ultima, se realiza un punto de guardado de la red
        network.eval()   ### Se activa la bandera de la red para que no entrene, de manera que lo que procese no afecta a los parametros
        # save checkpoint
        checkpoint(network, epoch, "model", opt.save_model_path, opt.log_name)   ### Se guarda el estado de la red
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       settings.systime)   ### Se guarda localmente la red neuronal
        validation_path = os.path.join(save_model_path, "validation")   ### Se crea la ruta a un nuevo directorio
        os.makedirs(validation_path, exist_ok=True)   ### Se crea un directorio de validacion
        np.random.seed(101)   ### Se genera una semilla
        valid_repeat_times = {"Kodak": 10, "BSD300": 3, "Set14": 20}   ### Se indica cuantas veces puede usarse cada dataset de validacion

        for valid_name, valid_images in valid_dict.items():   ### Para cada dataset de validacion
            psnr_result = []   ### Array para almacenar los psnr (Peak Signal Noise Ratio)
            ssim_result = []   ### Array para almacenar los ssim (Structural Similarity Index Measure)
            repeat_times = valid_repeat_times[valid_name]   ### Se obtiene cuantas veces va a procesarse cada dataset
            for i in range(repeat_times):   ### 
                for idx, im in enumerate(valid_images):   ### Para cada imagen del dataset actual
                    if opt.crop_size is not None: ### Se realiza un recorte aleatorio a la imagen si se indica un tamaño
                        im = crop_image(im, opt.crop_size, center_crop=True)
                    origin255 = im.copy()   ### Se crea una copia de la imagen original
                    origin255 = origin255.astype(np.uint8)   ### Se convierte a tipo entero sin signo
                    im = np.array(im, dtype=np.float32) / 255.0   ### Se convierte a numpy array de tipo float 32
                    noisy_im = noise_adder.add_valid_noise(im)   ### Añade ruido artificial a la imagen preparada
                    if epoch == opt.n_snapshot:   ### Si debe hacerse punto de guardado en esta epoca
                        noisy255 = noisy_im.copy()   ### Se hace una copia de la imagen ruidosa
                        noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                           255).astype(np.uint8)   ### Se desnormaliza la imagen, limitando los valores de los pixeles entre 0 y 255  (np.clip trunca los valores menores y mayores que el minimo y maximo indicado)
                    H = noisy_im.shape[0]   ### Se obtiene el alto de la imagen
                    W = noisy_im.shape[1]   ### Se obtiene el ancho de la imagen
                    if opt.crop_size is None: ### Si no es necesario recortar, se ajusta la imagen para que sea cuadrada
                        # padding to square
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

                    # visualization
                    if i == 0 and epoch == opt.n_snapshot:   ### Si es la primera imagen que se procesa y en la epoca actual se realiza punto de control
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_clean.png".format(
                                valid_name, idx, epoch))   ### Se crea el nombre de ruta para almacenar la imagen original
                        Image.fromarray(origin255).convert('RGB').save(
                            save_path)   ### Guarda en la ruta creada la imagen original
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_noisy.png".format(
                                valid_name, idx, epoch))   ### Se crea el nombre de ruta para almacenar la imagen con ruido
                        Image.fromarray(noisy255).convert('RGB').save(
                            save_path)   ### Guarda en la ruta creada la imagen original
                    if i == 0:   ### 
                        save_path = os.path.join(
                            validation_path,
                            "{}_{:03d}-{:03d}_denoised.png".format(
                                valid_name, idx, epoch))   ### Se crea el nombre de ruta para almacenar la imagen limpia generada
                        Image.fromarray(pred255).convert('RGB').save(save_path)   ### Guarda en la ruta creada la imagen original

            psnr_result = np.array(psnr_result)   ### Convierte el array de los psnr a tipo numpy
            avg_psnr = np.mean(psnr_result)   ### Obtiene la media del psnr de todas las imagenes evaluadas
            avg_ssim = np.mean(ssim_result)   ### Obtiene la media del ssim de todas las imagenes evaluadas
            log_path = os.path.join(validation_path,
                                    "A_log_{}.csv".format(valid_name))   ### Crea la ruta para guardar un fichero de registro
            with open(log_path, "a") as f:   ### Crea un flujo de escritura apuntando al final del fichero
                f.writelines("{},{},{}\n".format(epoch, avg_psnr, avg_ssim))   ### Escribe en el fichero la epoca y las medidas obtenidas
