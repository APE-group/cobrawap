"""
Deconvolutes the given signal by applying a digital filter with a given kernel function.
"""
import argparse
import quantities as pq
import os
import neo
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial import *
from scipy.optimize import curve_fit
from utils import load_neo, write_neo, none_or_str, save_plot, none_or_float, \
                  AnalogSignal2ImageSequence, ImageSequence2AnalogSignal

def gauss(x, mu, sigma, a):
    return a/(2 * np.pi * sigma**2)**(1/2) * np.exp(- (x - mu)**2 / (2 * sigma**2))

def start_point(quantile,y,res):
    for x in range(y.size):
        if np.sum(y[:x])*res < np.sum(y)*res*quantile:
            start = x
    return start

if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data",    nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    CLI.add_argument("--output",  nargs='?', type=str, required=True,
                     help="path of output file")
    CLI.add_argument("--bandwidth", nargs='?', type=str,
                     help="KDE bandwidth", default="silverman")
    CLI.add_argument("--quantile", nargs='?', type=float,
                     help="Quantile of the left outliers", default=0.05)
    # think about using alternatives to the gaussian kernel

    args = CLI.parse_args()

    # loads and converts the neo block
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    
    # converts the imagesequence class in a 3D Numpy array
    imgseq = block.segments[0].imagesequences[-1]
    imgseq_array = imgseq.as_array()
    dim_t, dim_x, dim_y = imgseq_array.shape

    KDE_results = np.zeros((5,dim_x,dim_y)) # mode (no error), mean and sigma with errors for every pixel
    sp_res = 1 # fixing the resolution in the activities space
    n_points = 20 #KDE fit sampling points
    """
    over 20 minutes of compiling time...
    for i in range(dim_x):
        for j in range(dim_y):
            kde_act = imgseq_array[:,i,j]
            kde = gaussian_kde(kde_act, bw_method=args.bandwidth)
            x_KDE = np.arange(kde_act.min(), kde_act.max(), sp_res) # apply the KDE on a grid
            ker_pdf = kde(x_KDE) # get the probability density function
            kde_mod = kde_act.min() + sp_res*ker_pdf.argmax() # mode as max(pdf)
            KDE_results[0,i,j] = kde_mod
            # focus on the left side of the distribution
            start = start_point(args.quantile, ker_pdf, sp_res)
            end = ker_pdf.argmax()
            xres = int(round((x_KDE[end]-x_KDE[start])/n_points)) #KDE sampling resolution
            xdata = x_KDE[start:end:xres]
            ydata = ker_pdf[start:end:xres]
            p0 = [kde_mod, kde_act.std(), 1] # fit starting parameters
            # fit, constraining the peak of the gaussian to be equal to the mode of the distribution
            popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0, bounds=([kde_mod-0.1, 0, 0], [kde_mod, np.inf, np.inf]))
            perr = np.sqrt(np.diag(pcov))
            KDE_results[1,i,j], KDE_results[2,i,j] = popt[0], popt[1]
            KDE_results[3,i,j], KDE_results[4,i,j] = perr[0], perr[1]
    """
    # focus on a single pixel
    i, j = 40, 80
    kde_act = imgseq_array[:,i,j]
    kde = gaussian_kde(kde_act, bw_method=args.bandwidth)
    x_KDE = np.arange(kde_act.min(), kde_act.max(), sp_res) # apply the KDE on a grid
    ker_pdf = kde(x_KDE) # get the probability density function
    kde_mod = kde_act.min() + sp_res*ker_pdf.argmax() # mode as max(pdf)
    KDE_results[0,i,j] = kde_mod
    # focus on the left side of the distribution
    start = start_point(args.quantile, ker_pdf, sp_res)
    end = ker_pdf.argmax()
    xres = int(round((x_KDE[end]-x_KDE[start])/n_points)) #KDE sampling resolution
    xdata = x_KDE[start:end:xres]
    ydata = ker_pdf[start:end:xres]
    p0 = [kde_mod, kde_act.std(), 1] # fit starting parameters
    # fit, constraining the peak of the gaussian to be equal to the mode of the distribution
    popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0, bounds=([kde_mod-0.1, 0, 0], [kde_mod, np.inf, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    KDE_results[1,i,j], KDE_results[2,i,j] = popt[0], popt[1]
    KDE_results[3,i,j], KDE_results[4,i,j] = perr[0], perr[1]

    print('Mode :', round(KDE_results[0,40,80]))
    print('Optimal mean :', round(KDE_results[1,40,80]), '+-', round(KDE_results[3,40,80]))
    print('Optimal sigma: ', round(KDE_results[2,40,80]), '+-', round(KDE_results[4,40,80]))

    write_neo(args.output, block)
