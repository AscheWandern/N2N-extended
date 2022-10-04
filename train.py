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
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from arch_unet import UNet

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

opt, _ = parser.parse_known_args()  ### Recopilar parametros de ejecucion
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')   ###  Formateo de fecha actual (ejecución)
operation_seed_counter = 0   ### Elección de semilla de aleatoriedad
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices   ### Selección de dispositivo gpu para la ejecucion

### Metodo para guardar el estado de la red
def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)   ### Crea la ruta donde se guardara el estado de la red
    os.makedirs(save_model_path, exist_ok=True)   ### Crea la carpeta si no está creada
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)   ### Establece el nombre del archivo con el modelo
    save_model_path = os.path.join(save_model_path, model_name)   ### Crea la ruta completa del archivo que se creará
    torch.save(net.state_dict(), save_model_path)   ### Guarda los pesos del modelo
    print('Checkpoint saved to {}'.format(save_model_path))

### Modifica la semilla del generador de numeros aleatorios y devuelve un generador con la semilla correspondiente
def get_generator():
    global operation_seed_counter   ### Accede a la variable externa para modificarla
    operation_seed_counter += 1   ### Incrementa la semilla
    g_cuda_generator = torch.Generator(device="cuda")   ### Crea un generador de números aleatorios
    g_cuda_generator.manual_seed(operation_seed_counter)   ### Establece la semilla con la creada anteriormente
    return g_cuda_generator

### Clase destinada a la inclusión de ruido artificial a las imágenes de entrada
class AugmentNoise(object):
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):   ### Si el ruido elegido es de tipo gaussiano
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')   ### Normaliza los valores de ruido indicados
            ]
            if len(self.params) == 1:   ### Si solo hay un elemento el valor es fijo
                self.style = "gauss_fix"
            elif len(self.params) == 2:   ### Si hay dos elementos se indica un rango
                self.style = "gauss_range"
        elif style.startswith('poisson'):   ### Si el ruido elegido es de tipo poisson
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')   ### Normaliza los valores de ruido indicados
            ]
            if len(self.params) == 1:   ### Si solo hay un elemento el valor es fijo
                self.style = "poisson_fix"
            elif len(self.params) == 2:   ### Si hay dos elementos se indica un rango
                self.style = "poisson_range"

    def add_train_noise(self, x):   ### Metodo para incluir ruido en las imagenes de entrenamiento
        shape = x.shape   ### Guarda las dimensiones de la imagen
        if self.style == "gauss_fix":   ### Tipo de ruido es gaussiano de valor fijo
            std = self.params[0]   ### Guarda la desviación estandar que se usará
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)   ### Crea una matriz rellena del valor estándar en el el dispositivo de ejecucion
            noise = torch.cuda.FloatTensor(shape, device=x.device)   ### Crea un tensor del tamaño de la imagen de tipo float para almacenar el ruido
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)   ### Genera el ruido normalizado con un generador usando la matriz de valor estandar creada anteriormente
            return x + noise   ### Devuelve la suma de la imagen y la matriz de ruido para crear la imagen con ruido
        elif self.style == "gauss_range":   ### Tipo de ruido es gaussiano con valor variable
            min_std, max_std = self.params   ### Guarda la desviacion minima y maxima
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std   ### Genera una matriz de valores aleatorios entre el maximo y el minimo de variacion
            noise = torch.cuda.FloatTensor(shape, device=x.device)   ### Crea un tensor del tamaño de la imagen de tipo float para almacenar el ruido
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)   ### Genera el ruido normalizado con un generador usando la matriz de valor estandar creada anteriormente
            return x + noise   ### Devuelve la suma de la imagen y la matriz de ruido para crear la imagen con ruido
        elif self.style == "poisson_fix":   ### Tipo de ruido es de poisson de valor fijo
            lam = self.params[0]   ### Guarda el indice de ruido
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)   ### Genera una matriz rellena con el valor del ruido
            noised = torch.poisson(lam * x, generator=get_generator()) / lam   ### Genera un tensor con distribucion de Poisson normalizado
            return noised   ### Devuelve la imagen con el ruido ya aplicado
        elif self.style == "poisson_range":   ### Tipo de ruido es de poisson con valor variable
            min_lam, max_lam = self.params   ### Guarda el indice minimo y maximo de ruido de Poisson
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam   ### Genera una matriz con valores aleatorios entre los valores maximos y minimos
            noised = torch.poisson(lam * x, generator=get_generator()) / lam   ### Genera un tensor con distribucion de Poisson normalizado
            return noised   ### Devuelve la imagen con el ruido ya aplicado

    def add_valid_noise(self, x):   ### Metodo para incluir ruido de validacion
        shape = x.shape   ### Guarda las dimensiones de la imagen
        if self.style == "gauss_fix":   ### Tipo de ruido gaussiano fijo
            std = self.params[0]   ### Guarda la desviacion estandar de ruido
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)   ### Devuelve la imagen con ruido incluido (imagen mas vector normalizado por nivel de ruido)
        elif self.style == "gauss_range":   ### Tipo de ruido gaussiano variable
            min_std, max_std = self.params   ### Guarda los valores de desviacion minimo y maximo
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))   ### Genera un vector de valores normalizados aleatorios entre los valores dados
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)   ### Devuelve la imagen con ruido incluido (imagen mas vector normalizado por nivel de ruido)
        elif self.style == "poisson_fix":   ### Tipo de ruido poisson fijo
            lam = self.params[0]   ### Guarda el indice de ruido
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)   ### Devuelve la imagen con ruido normalizada
        elif self.style == "poisson_range":   ### Tipo de ruido poisson variable
            min_lam, max_lam = self.params   ### Guarda los indices de ruido minimo y maximo
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))   ### Genera un vector de valores normalizados aleatorios entre los valores dados
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)   ### Devuelve la imagen con ruido normalizada


def space_to_depth(x, block_size):
    n, c, h, w = x.size()   ### Guarda el numero de muestras, canales de la imagen y su alto y ancho
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)   ### 
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)   ### 


def generate_mask_pair(img):   ### Genera las mascaras de pixeles que se usaran para generan las subimagenes
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape   ### Guarda el numero de muestras, canales de la imagen y su alto y ancho
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)   ### Genera la máscara para la primera imagen inicializada a 0
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)   ### Genera la máscara para la segunda imagen inicializada a 0
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)   ### Crea un tensor con los pares de indices que indican pixeles vecinos
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)   ### Crea un tensor de ceros cuyo tamaño es la mitad que la imagen original
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)   ### Rellena el tensor de ceros con numeros aleatorios entre 0 y 8
    rd_pair_idx = idx_pair[rd_idx]   ### Selecciona los indices de pares de pixeles vecinos de cada bloque para posteriormente crear las imagenes
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)   ### Suma a los índices un vector de posiciones incremental de 4 en 4 (cada bloque son 4 pixeles, 2x2)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1   ### Activa las posiciones de la máscara correspondientes a los píxeles de la primera subimagen
    mask2[rd_pair_idx[:, 1]] = 1   ### Activa las posiciones de la máscara correspondientes a los píxeles de la segunda subimagen
    return mask1, mask2   ### Retorna las dos mascaras para crear las subimagenes de pixeles vecinos


def generate_subimages(img, mask):   ### Genera una subimagen a partir de otra dada y la mascara de pixeles
    n, c, h, w = img.shape   ### Guarda el numero de muestras, canales de la imagen y su alto y ancho
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)   ### Crea un tensor inicializado a ceros con sus dimensiones siendo la mitad de la imagen original
    # per channel
    for i in range(c):   ### Para cada canal de la nueva imagen se copian los pixeles de la imagen original que indica la mascara
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)   ### Se obtiene el canal que indica el indice en bloques de 2x2 (esto es necesario porque las imagenes se van a dividir a la mitad)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)   ### Cambia las capas de orden y las convierte a array
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)   ### Reconvierte el array a capa de canal y asigna la capa a la nueva imagen
    return subimage   ### Retorna la subimagen ya creada


class DataLoader_Imagenet_val(Dataset):   ### Clase dedicada a la carga del dataset
    def __init__(self, data_dir, patch=256):   ### Metodo de inicializacion del cargador
        super(DataLoader_Imagenet_val, self).__init__()   ### Inicializador de la clase padre
        self.data_dir = data_dir   ### Guarda el directorio de datos
        self.patch = patch   ### Guarda el tamaño de parche (dimensiones finales)
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))   ### Almacena nombres de imagenes del dataset
        self.train_fns.sort()   ### Ordena las imagenes del dataset
        print('fetch {} samples for training'.format(len(self.train_fns))) ### Muestra el numero de imagenes para procesar

    def __getitem__(self, index):   ### Extrae parches de la imagen para la entrada de la red
        # fetch image
        fn = self.train_fns[index]   ### Elige la imagen en base al indice dado
        im = Image.open(fn)   ### Carga la imagen
        im = np.array(im, dtype=np.float32)   ### Convierte la imagen a un numpy array de tipo float 
        # random crop
        H = im.shape[0]   ### Obtiene la altura de la imagen
        W = im.shape[1]   ### Obtiene el ancho de la imagen
        if H - self.patch > 0:   ### Si el alto es mas grande que el tamaño maximo, se recorta
            xx = np.random.randint(0, H - self.patch)   ### Obtiene una posicion aleatoria donde comenzara la altura del parche
            im = im[xx:xx + self.patch, :, :]   ### Obtiene la imagen a partir de la posicion indicada (recorta el alto)
        if W - self.patch > 0:   ### Si el ancho es mas grande que el tamaño maximo, se recorta
            yy = np.random.randint(0, W - self.patch)   ### Obtiene una posicion aleatoria donde comenzara la anchura del parche
            im = im[:, yy:yy + self.patch, :]   ### Obtiene la imagen a partir de la posicion indicada (recorta el ancho)
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])   ### Prepara un transformador a tensor
        im = transformer(im)   ### Convierte la imagen recortada en un tensor
        return im   ### Devuelve la imagen final en forma de tensor

    def __len__(self):   ### Metodo propio de la clase para indicar la longitud de los datos
        return len(self.train_fns)   ### Devuelve la cantidad de imagenes que hay en el dataset cargado


def validation_kodak(dataset_dir):   ###  Metodo para cargar el dataset kodak
    fns = glob.glob(os.path.join(dataset_dir, "*"))   ### Carga de los nombres del directorio
    fns.sort()   ### Ordenar nombres del directorio
    images = []   ### Creacion de array para las imagenes ya cargadas
    for fn in fns:   ### Para cada imagen del directorio
        im = Image.open(fn)   ### Carga la imagen
        im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
        images.append(im)   ### Añade la imagen al array
    return images   ### Devuelve el array de imagenes ya cargadas


def validation_bsd300(dataset_dir):   ### Metodo para cargar el dataset BSD300
    fns = []   ### Crea un array para los archivos del directorio
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))   ### Carga los archivos en el array 
    fns.sort()   ### Ordena los nombres de los archivos
    images = []   ### Crea un array para las imagenes
    for fn in fns:   ### Para cada archivo en el directorio
        im = Image.open(fn)   ### Carga la imagen
        im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
        images.append(im)   ### Añade la imagen al array
    return images   ### Devuelve el array de imagenes


def validation_Set14(dataset_dir):   ### Metodo para cargar el dataset Set14
    fns = glob.glob(os.path.join(dataset_dir, "*"))   ### Carga los nombres del directorio
    fns.sort()   ### Ordena los nombres de las imagenes
    images = []   ### Crea un array para las imagenes
    for fn in fns:   ### Por cada archivo del directorio
        im = Image.open(fn)   ### Carga la imagen
        im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
        images.append(im)   ### Añade la imagen al array
    return images   ### Devuelve el array de imagenes ya cargadas


def ssim(prediction, target):   ### Metodo para calcular el ssim (structural similarity index measure) entre dos imagenes
    ##### Se calculan las dos constantes estabilizadoras de la division
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    ### Se transforman las imagenes a tipo float de 64 bits
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)   ### Se crea un kernel de 11x11 (radio de 5 desde el centro) con desviacion 1.5
    ### Multiplica el kernel por si mismo traspuesto (la matriz resultado es simetrica)
    window = np.outer(kernel, kernel.transpose())
    
    #mu es la media de la muestra de pixeles de la imagen y sigma square la varianca de la imagen
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    ##### Se obtienen las versiones cuadradas de ambos parametros y su producto para futuros calculos
    mu1_sq = mu1**2 
    mu2_sq = mu2**2 
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq   ### 
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq   ### 
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2   ### 
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))   ### Obtiene el mapa de ssim (se esta trabajando con matrices, con lo que es el conjunto de ssim de las ventanas obtenidas de las imagenes)
    return ssim_map.mean()   ### Se calcula la media de los ssim de cada ventana obtenida


def calculate_ssim(target, ref):   ### Metodo para preparar las opciones y calcular posteriormente el ssim
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    ### Se convierten ambas imagenes a tipo float de 64 bits
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64) 
    if not img1.shape == img2.shape:   ### Comprueba que las imagenes poseen el mismo tamaño
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:   ### Si la imagen es en blanco y negro (alto x ancho)
        return ssim(img1, img2)   ### Devuelve el ssim
    elif img1.ndim == 3:   ### Si la imagen tiene mas canales, por ejemplo a color (alto x ancho x canales)
        if img1.shape[2] == 3:   ### Si son tres canales
            ssims = []   ### Se crea un array para almacenar los ssim de cada uno
            for i in range(3):   ### Para cada canal
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))   ### Se obtiene el ssim de los respectivos canales de ambas imagenes
            return np.array(ssims).mean()   ### Se devuelve la media de los ssim
        elif img1.shape[2] == 1:   ### Si la dimension de los canales es 1 (es blanco y negro pero se ha cargado de esta manera)
            return ssim(np.squeeze(img1), np.squeeze(img2))   ### Comprime las imagenes a 2 dimensiones y calcula su psnr
    else:   ### En cualquier otro caso hay un error con la imagen y salta un error
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):   ### Metodo para calcular el psnr
    ### Se convierten ambas imagenes a tipo float de 32 bits
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2   ### Se calcula la diferencia entre las imagenes
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))   ### Aplica la formula del psnr
    ##### La media del cuadrado de la diferencia es lo mismo que calcular el MSE (Mean Square Error) entre las imagenes
    ##### 255.0 * 255.0 es lo mismo que MAX², ya que 255 es el maximo valor posible para un pixel
    return psnr   ### Devuelve el psnr calculado


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
valid_dict = {
    "Kodak": validation_kodak(Kodak_dir),
    "BSD300": validation_bsd300(BSD300_dir),
    "Set14": validation_Set14(Set14_dir)
}   ### Creacion de un diccionario para la obtencion de los datasets

# Noise adder
noise_adder = AugmentNoise(style=opt.noisetype)   ### Creacion de objeto que servira para la inclusion de ruido artificial

# Network
network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)   ### Creación de la red neuronal
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

checkpoint(network, 0, "model")   ### Creacion de un punto de guardado del modelo antes de entrenar
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
        checkpoint(network, epoch, "model")   ### Se guarda el estado de la red
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)   ### Se guarda localmente la red neuronal
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
