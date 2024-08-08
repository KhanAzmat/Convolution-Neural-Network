import pickle
import matplotlib.pyplot as plt

# to read from pickle file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# to reshape and transpose the images
def reshape_transpose(data):
    data = data.reshape(len(data), 3,32,32)
    return data.transpose(0,2,3,1)

# Visualize some images 
def visualise(data):   
    fig = plt.figure(figsize=(5,5))

    for i in range(6):
        fig.add_subplot(3, 3, i+1)
        plt.imshow(data[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def normalise(data):
    return data.astype('float32')/255.0
