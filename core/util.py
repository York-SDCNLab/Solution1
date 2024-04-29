import matplotlib.pyplot as plt
from multiprocessing import Queue 

def plot_line_chart(data, x_label='x_label', y_label='y_label', title="linear chart") -> None: 
    plt.figure()
    plt.plot(data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def handle_interprocess(data_queue, item, block=True, timeout=None) -> None: 
    try: 
        data_queue.put(item, block, timeout)
    except Queue.Full: 
        data_queue.get_nowait() 
        data_queue.put(item, block, timeout)