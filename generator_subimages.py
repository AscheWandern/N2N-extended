import torch
from utils import get_generator

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
