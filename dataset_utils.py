import numpy as np
import torch
import glob
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


from utils import get_generator

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
            std = std * torch.ones(shape, device=x.device)   ### Crea una matriz rellena del valor estándar en el el dispositivo de ejecucion
            ### Crea un tensor del tamaño de la imagen de tipo float para almacenar el ruido
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)   ### Genera el ruido normalizado con un generador usando la matriz de valor estandar creada anteriormente
            return x + noise   ### Devuelve la suma de la imagen y la matriz de ruido para crear la imagen con ruido
        elif self.style == "gauss_range":   ### Tipo de ruido es gaussiano con valor variable
            min_std, max_std = self.params   ### Guarda la desviacion minima y maxima
            std = torch.rand(size=shape,
                             device=x.device) * (max_std - min_std) + min_std   ### Genera una matriz de valores aleatorios entre el maximo y el minimo de variacion
            ### Crea un tensor del tamaño de la imagen de tipo float para almacenar el ruido
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)   ### Genera el ruido normalizado con un generador usando la matriz de valor estandar creada anteriormente
            return x + noise   ### Devuelve la suma de la imagen y la matriz de ruido para crear la imagen con ruido
        elif self.style == "poisson_fix":   ### Tipo de ruido es de poisson de valor fijo
            lam = self.params[0]   ### Guarda el indice de ruido
            lam = lam * torch.ones(shape, device=x.device)   ### Genera una matriz rellena con el valor del ruido
            noised = torch.poisson(lam * x, generator=get_generator()) / lam   ### Genera un tensor con distribucion de Poisson normalizado
            return noised   ### Devuelve la imagen con el ruido ya aplicado
        elif self.style == "poisson_range":   ### Tipo de ruido es de poisson con valor variable
            min_lam, max_lam = self.params   ### Guarda el indice minimo y maximo de ruido de Poisson
            lam = torch.rand(size=shape,
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

### Clase dedicada a la carga del dataset
class DataLoader_Imagenet_val(Dataset):   
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
        im = crop_image(im, self.patch)
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])   ### Prepara un transformador a tensor
        im = transformer(im)   ### Convierte la imagen recortada en un tensor
        return im   ### Devuelve la imagen final en forma de tensor

    def __len__(self):   ### Metodo propio de la clase para indicar la longitud de los datos
        return len(self.train_fns)   ### Devuelve la cantidad de imagenes que hay en el dataset cargado


def crop_image(image, patch=256, center_crop=False): ### Crea un recorte aleatorio de la imagen con limite de dimension indicado por el parametro
    H = image.shape[0]   ### Obtiene la altura de la imagen
    W = image.shape[1]   ### Obtiene el ancho de la imagen
    if H - patch > 0:   ### Si el alto es mas grande que el tamaño maximo, se recorta
        if center_crop:
            xx = H//2 - patch//2
        else:
            xx = np.random.randint(0, H - patch)   ### Obtiene una posicion aleatoria donde comenzara la altura del parche
        image = image[xx:xx + patch, :, :]   ### Obtiene la imagen a partir de la posicion indicada (recorta el alto)
    if W - patch > 0:   ### Si el ancho es mas grande que el tamaño maximo, se recorta
        if center_crop:
            yy = W//2 - patch//2
        else:
            yy = np.random.randint(0, W - patch)   ### Obtiene una posicion aleatoria donde comenzara la anchura del parche
        image = image[:, yy:yy + patch, :]   ### Obtiene la imagen a partir de la posicion indicada (recorta el ancho)
    return image


class DataLoader_Validation(object):
    def __init__(self, kodak_dir, bsd300_dir, set14_dir):   
        self.kodak_dir = kodak_dir
        self.bsd300_dir = bsd300_dir
        self.set14_dir = set14_dir
        
    def kodak(self):   ###  Metodo para cargar el dataset kodak
        fns = glob.glob(os.path.join(self.kodak_dir, "*"))   ### Carga de los nombres del directorio
        fns.sort()   ### Ordenar nombres del directorio
        images = []   ### Creacion de array para las imagenes ya cargadas
        for fn in fns:   ### Para cada imagen del directorio
            im = Image.open(fn)   ### Carga la imagen
            im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
            images.append(im)   ### Añade la imagen al array
        return images   ### Devuelve el array de imagenes ya cargadas


    def bsd300(self):   ### Metodo para cargar el dataset BSD300
        fns = []   ### Crea un array para los archivos del directorio
        fns.extend(glob.glob(os.path.join(self.bsd300_dir, "test", "*")))   ### Carga los archivos en el array 
        fns.sort()   ### Ordena los nombres de los archivos
        images = []   ### Crea un array para las imagenes
        for fn in fns:   ### Para cada archivo en el directorio
            im = Image.open(fn)   ### Carga la imagen
            im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
            images.append(im)   ### Añade la imagen al array
        return images   ### Devuelve el array de imagenes


    def set14(self):   ### Metodo para cargar el dataset Set14
        fns = glob.glob(os.path.join(self.set14_dir, "*"))   ### Carga los nombres del directorio
        fns.sort()   ### Ordena los nombres de las imagenes
        images = []   ### Crea un array para las imagenes
        for fn in fns:   ### Por cada archivo del directorio
            im = Image.open(fn)   ### Carga la imagen
            im = np.array(im, dtype=np.float32)   ### Transforma la imagen a numpy array de tipo float 
            images.append(im)   ### Añade la imagen al array
        return images   ### Devuelve el array de imagenes ya cargadas
