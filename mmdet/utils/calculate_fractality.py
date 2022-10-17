import sys
sys.path.insert(1, '../') 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from multiprocessing import Pool
from scipy import optimize
import argparse


def main(args):
    gt = pd.read_csv(args.path)
    dset_name = args.dset_name
    class_names = []
    global calculate_fract
    if dset_name =='coco':
        num_classes = 90
        with open('./lvis_visualisations/coco_names.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
    elif dset_name =='lvisv1':
        num_classes = 1204
        with open('./lvis_visualisations/lvis_names.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
    else:
        num_classes = 1230
        with open('./lvis_visualisations/lvis_names_v0.5.txt','r') as file:
            for name in file.readlines():
                class_names.append(name.rstrip())
        
    cx=np.array(gt['xmin'])+np.array(gt['width'])/2
    cy = np.array(gt['ymin'])+np.array(gt['height'])/2
    gt_categories=np.array(gt['category'])-1
    dimensions = np.arange(args.dims)+1
    frequencies=np.bincount(gt_categories,minlength=num_classes)

    def calculate_fract(dim,cx=cx,cy=cy,gt_categories=gt_categories,num_classes=num_classes,dataset=dset_name):
        frequency=np.bincount(gt_categories,minlength=num_classes)
        step = 1/dim
        if dataset == 'coco':
            coco91_mask = frequency!=0
            gt_loc_bias=np.zeros((dim,dim,80))
        else:
            gt_loc_bias=np.zeros((dim,dim,num_classes))
        gt_img=np.zeros((dim,dim))

        for j in range(dim):
            for i in range(dim):
                dimx=[i*step,(i+1)*step]
                maskx= (cx>=dimx[0])&(cx<dimx[1])
                dimy=[j*step,(j+1)*step]
                masky= (cy>=dimy[0])&(cy<dimy[1])
                mask_final=maskx&masky
                gt_img[j,i]=mask_final.sum()
                g=gt_categories[mask_final]
                bins=np.bincount(g,minlength=num_classes)
                if dataset == 'coco':
                    gt_loc_bias[j,i,:] = (bins[coco91_mask])
                else:
                    gt_loc_bias[j,i,:] = bins
        boxes = (gt_loc_bias>0).sum(axis=0).sum(axis=0)

        return boxes
    
    p = Pool(args.workers)
    grid_size = args.dims
    dims = np.arange(grid_size)+1
    boxes=p.map(calculate_fract, dims)
    
    if dset_name =='coco':
        num_classes = 80
    

    def fit(x, A, Df):
        """
        User defined function for scipy.optimize.curve_fit(),
        which will find optimal values for A and Df.
        """
        return Df * x + A
    
    fractality =  []
    for k in range(num_classes):
        N = [boxes[i][k] for i in np.arange(grid_size)]
        cuttof_ponts = (frequencies[k])>=dims*dims
        if frequencies[k]<4:
            fractality.append(1)
        else:
            popt, pcov =optimize.curve_fit(fit, np.log(dims)[cuttof_ponts], np.log(N)[cuttof_ponts], maxfev=100000)
            if popt[1]<1:
                fractality.append(1)
            else:
                fractality.append(popt[1])
                
    df = pd.DataFrame()
    df['class_names'] = class_names
    df['fractal_dimension'] = fractality
    
    df.to_csv(args.output)
            



    

def get_args_parser():
    parser = argparse.ArgumentParser(description='Parse arguments for per shot acc.')
    
    parser.add_argument('--dset_name', default='lvis05',type=str, help='Dataset Name lvis|coco')
    parser.add_argument(
        '--path', default='./lvis_statistics_val_v0.5.csv', help='path to statistics')
    parser.add_argument(
        '--output', default='./output.csv', help='output file',type=str)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',help='number of workers (default: 32)')
    parser.add_argument('-d', '--dims', default=64, type=int, metavar='N',help='grid max dimension')

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
    print('end of program')