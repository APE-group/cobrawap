"""
Downsampling: change the image resolution creating macropixels of chosen size, each one with the mean value of the original pixels in it.
"""
import argparse
import quantities as pq
import os
import numpy as np
import matplotlib.pyplot as plt
import neo
from utils import load_neo, write_neo, AnalogSignal2ImageSequence,  ImageSequence2AnalogSignal, save_plot, none_or_str


def plot_down(img):
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    return ax

def downsampling(img, orig_res, new_res, choice):
    n = round((new_res/orig_res).magnitude.tolist()) # number of pixels in a side of the macropixel
    orig_side = np.shape(imgseq_array[0])[0] # dimensione lato griglia = 100
    new_dim = n**(np.ndim(imgseq_array[0])) # n^(dimensioni griglia=2)
    new_side = int(orig_side/n) # dimensione nuovo lato griglia, prende l'intero più basso
    tmax = np.shape(imgseq_array)[0] # numero totale di frame = 1000
    down = np.zeros((tmax, new_side, new_side)) # nuova matrice 3D di numpy con i dati di downsampling

    if choice == 'y':
        for t in range(tmax):
            for i in range(new_side):
                for j in range(new_side):
                    z = 0 
                    for k in range(n):
                        for l in range(n):
                            z += imgseq_array[t][n*i + k][n*j + l]
                    down[t][i][j] = z/new_dim

    else:
        if n%2==0:
            for t in range(tmax): # ciclo sui frame
                for i in range(new_side):
                    for j in range(new_side):
                        down[t][i][j] = imgseq_array[t][i*n+int(n/2)][j*n+int(n/2)]

        else:
            for t in range(tmax): # ciclo sui frame
                for i in range(new_side):
                    for j in range(new_side):
                        down[t][i][j] = imgseq_array[t][i*n+int(n/2)+1][j*n+int(n/2)+1]

    return down


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--output_img",  nargs='?', type=none_or_str,
                     help="path of output image", default=None)
    CLI.add_argument("--new_resolution", nargs='?', type=float,
                     help="new spatial resolution in mm", default=0.10)
    CLI.add_argument("--mean",  nargs='?', type=none_or_str,
                     help="mean on pixels? (y/n)", default="n")
    args = CLI.parse_args()

    # loads and converts the neo block
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    
    # converts the imagesequence class in a 3D Numpy array
    imgseq = block.segments[0].imagesequences[-1]
    imgseq_array = imgseq.as_array()
    
    # getting the two reaolutions
    orig_res = block.segments[0].imagesequences[0].spatial_scale
    new_res = (args.new_resolution) * pq.mm
    n = round((new_res/orig_res).magnitude.tolist())
    # check on the < resolution
    if new_res.magnitude.tolist() < orig_res.magnitude.tolist(): 
        print("ERROR - New spatial scale must be greater than the original one. \n")
    
    # downsampling
    downsampled = downsampling(imgseq_array, orig_res, new_res, args.mean)
    dim_t, dim_x, dim_y = downsampled.shape

    """
    Old method, it doesn't work properly (the ouput block doesn't work as input for the other pipeline modules)
    
    # re-converting into analogsignal
    signal = downsampled.reshape((dim_t, dim_x * dim_y))
    asig = block.segments[0].analogsignals[0].duplicate_with_new_data(signal)
    #asig.array_annotate(**block.segments[0].analogsignals[0].array_annotations)

    # save data and figure
    asig.name += ""
    asig.description += "Downsampled image with resolution of {} ({})"\
                        .format(args.new_resolution, os.path.basename(__file__))
    block.segments[0].analogsignals[0] = asig
    """
    
    # New method, creating a new block
    # create an ImageSequence with the downsampled matrix
    imgseq_down = neo.ImageSequence(imgseq_array, units = block.segments[0].analogsignals[0].units, 
                      sampling_rate = block.segments[0].analogsignals[0].sampling_rate, 
                      spatial_scale = orig_res*n, name = block.segments[0].analogsignals[0].name, 
                      description = block.segments[0].analogsignals[0].description)
    
    # create a new Block & Segment and append the ImageSequence
    segm_down = neo.Segment()
    segm_down.imagesequences.append(imgseq_down)
    block_down = neo.Block()
    block_down.segments.append(segm_down)
    block_down.name = block.name
    block_down.description = block.description
    block_down.annotations = block.annotations
    
    # converting into analogsignal 
    block_down = ImageSequence2AnalogSignal(block_down)
    
    # plot images
    plot_down(downsampled[0])
    save_plot(args.output_img)

    write_neo(args.output, block_down)