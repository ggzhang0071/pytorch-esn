import argparse
import  ESNRegularizationCompare as EGC

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--Monte_carlo', type=int, required=True,default=20,
                    help='Monte Carlo Iterations')

parser.add_argument('--savepath', type=str,required=False, default='../Results/',
                    help='Path to save results')
args = parser.parse_args()
EGC.ESNRegularizationCompare(args.Monte_carlo,args.savepath)