import argparse
import  PSOGSA_ESN as PSN

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--max_iters', type=int, required=True,default=50,
                    help='')
parser.add_argument('--num_particles', type=int,required=True, default=30,
                    help='')
parser.add_argument('--savepath', type=str,required=False, default='../Results/',
                    help='Path to save results')
args = parser.parse_args()
PSN.PSOGSAESN(args.max_iters,args.num_particles,args.savepath)