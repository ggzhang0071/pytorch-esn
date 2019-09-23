timestamp=`date +%Y%m%d%H%M%S`
dataset='solar-energy'
python -m pdb PSOGSA_ESN.py  --dataset $dataset --ngpu 4 --max_iters 50  --num_particles 30  |tee PSOGSA_ESN_$dataset_$timestamp.log


dataset='traffice'
python PSOGSA_ESN.py  --dataset $dataset --ngpu 4  --max_iters 50  --num_particles 30  |tee PSOGSA_ESN_$dataset_$timestamp.lo