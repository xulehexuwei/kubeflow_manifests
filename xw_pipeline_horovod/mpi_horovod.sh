mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py




mpirun -np 2 --allow-run-as-root -bind-to none -map-by slot  -mca pml ob1 -mca btl ^openib python /examples/tensorflow_mnist.py --lr 0.01 --num-steps 1000