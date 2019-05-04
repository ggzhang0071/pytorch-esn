TOP=`pwd`/..
timestamp=`date +%Y%m%d%H%M%S`
cl=`git rev-parse HEAD|cut -c1-7`

cd $TOP/examples

rm -fr *.log
python3 PSOGSA_ESN.py 2>&1 |tee esn_${cl}_$timestamp.log
#python3 ESNRegularizationCompare.py 2>&1 |tee esn_regularizationCompare_${cl}_$timestamp.log


