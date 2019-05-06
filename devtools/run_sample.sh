TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`

cd $TOP/examples

rm -fr *.log
python  MainPSOGSA_ESN.py  --max_iters 50 --num_particles 30 --savepath '../Results/' 2>&1 |tee esn_${cl}_$timestamp.log
python MainESNRegularizationCompare.py --Monte_carlo 1 --savepath '../Results/' 2>&1 |tee esn_regularizationCompare_${cl}_$timestamp.log


