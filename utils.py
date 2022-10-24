import torch
import os
import settings

### Modifica la semilla del generador de numeros aleatorios y devuelve un generador con la semilla correspondiente
def get_generator():
    settings.operation_seed_counter += 1  ### Accede a la variable externa para modificarla e incrementa la semilla
    g_cuda_generator = torch.Generator(device="cuda")   ### Crea un generador de números aleatorios
    g_cuda_generator.manual_seed(settings.operation_seed_counter)   ### Establece la semilla con la creada anteriormente
    return g_cuda_generator


### Metodo para guardar el estado de la red
def checkpoint(net, epoch, name, path, log_name):
    save_model_path = os.path.join(path, log_name, settings.systime)   ### Crea la ruta donde se guardara el estado de la red
    os.makedirs(save_model_path, exist_ok=True)   ### Crea la carpeta si no está creada
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)   ### Establece el nombre del archivo con el modelo
    save_model_path = os.path.join(save_model_path, model_name)   ### Crea la ruta completa del archivo que se creará
    torch.save(net.state_dict(), save_model_path)   ### Guarda los pesos del modelo
    print('Checkpoint saved to {}'.format(save_model_path))
