import numpy as np
import cv2

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
