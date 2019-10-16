export PYTHONPATH=$PYTHONPATH:`pwd`/src
export PYTHONPATH=$PYTHONPATH:`pwd`/src/bert
#export CUDA_VISIBLE_DEVICES=1

source ~/Code/venv_python36/bin/activate
export CUDA_VISIBLE_DEVICES=0

nohup python3.6 src/model/nli_main.py	\
	--output_dir=./out_1/\
	--data_dir=./dat/\
    --learning_rate=0.00003\
    --num_train_epochs=10.0\
	--vocab_file=../../ckpt_us/vocab.txt\
	--config_file=../../ckpt_us/bert_config.json\
    --init_checkpoint=../../ckpt_us/bert_model.ckpt\
	--do_eval=True\
	--do_train=False > log_eval.txt 2>&1 &

tail -f log_eval.txt
