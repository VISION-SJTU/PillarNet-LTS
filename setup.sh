cd det3d/ops/iou3d_nms
python setup.py build_ext --inplace

cd .. && cd  roiaware_pool3d
python setup.py build_ext --inplace

cd .. && cd  pillar_ops
python setup.py build_ext --inplace
