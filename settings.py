import datetime

def init():
    global operation_seed_counter, systime, torch_seed
    operation_seed_counter = 0
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')   ###  Formateo de fecha actual (ejecución)
    torch_seed = 0
