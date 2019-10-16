import sentencepiece as spm

file_list = [
	"iwslt2016/train"
]
vocab_size = 8000
model_type = 'bpe'

arg_str = "--input=%s --model_prefix=m --vocab_size=%d --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 --model_type=%s" \
					% (','.join(file_list), vocab_size,	 model_type)
print(arg_str)
spm.SentencePieceTrainer.Train(arg_str)
