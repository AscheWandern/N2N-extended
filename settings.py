import datetime

def init():
    global operation_seed_counter, systime
    operation_seed_counter = 0
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')   ###  Formateo de fecha actual (ejecución)
