wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1x6Noj88di4Cj8m4Gz7SoNaWkTNYnIFLJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1x6Noj88di4Cj8m4Gz7SoNaWkTNYnIFLJ" -O ./saved_models/model_69.ckpt  && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xXlkTwr1NZU_KhB7XZf4KmXeBLB2SJoY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xXlkTwr1NZU_KhB7XZf4KmXeBLB2SJoY" -O ./saved_models/model_69.ckpt.meta  && rm -rf /tmp/cookies.txt

python seq_to_seq_test.py $1 $2

python bleu_eval.py $2
