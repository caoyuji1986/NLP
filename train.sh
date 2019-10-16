export PYTHONPATH=$PYTHONPATH:./src
export CUDA_VISIBLE_DEVICES=0
nohup python3.6 src/nmt/transformer/train.py --batch_size=32 --num_train_samples=4468840 > log_train.txt 2>&1 &
