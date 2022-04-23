#!/usr/bin/python3
import argparse
from email.mime import base
from locale import normalize
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtrans
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import math
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   :12}

#matplotlib.rc('font', **font)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12 })


matplotlib.rcParams['agg.path.chunksize'] = 10000 
yaxis_label_local = {"cublas": "TFLOPS",
               "stream": "GB/s",
               "CUFFT" : "TFLOPs",
               "HPCG" : "TFLOPs",
               "miniFE" : "TFLOPs",
               "Cloverleaf" : "Cells/Second",
               "dgemm":"TFLOPS  "}
# Expect cpu, gpu input files in that order
def main():
    is_cpu=True
    parser = argparse.ArgumentParser(description="plot one text")
    parser.add_argument('files',  type=str,nargs="+", help='input file path')
    parser.add_argument('-c',  type=int, default=0, help='Critical Device 0 for Processor and 1 for Memory')
    args = parser.parse_args()
    critical_device=args.c
    data=[]
    for file in args.files:
    
        data.append(pd.read_csv(file))
    plt.plot(data[0]['Iteration'],data[0]['Inception Score'],label="DCGAN")
    plt.plot(data[1]['Iteration'],data[1]['Inception Score'],label="WCGAN")
    plt.plot(data[2]['Iteration'],data[2]['Inception Score'],label="ACGAN")

    plt.xlabel("Number Of Iterations")
    plt.ylabel("Inception Score")
    plt.legend()
    plt.savefig("data.png")

if __name__ == "__main__":
    main()
