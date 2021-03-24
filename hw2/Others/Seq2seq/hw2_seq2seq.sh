wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h5sMPbpDXrJxkdY7SUJ2hDUSYo3U1zXe' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h5sMPbpDXrJxkdY7SUJ2hDUSYo3U1zXe" -O ./saved_models/model_59.ckpt  && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1m48NKmXKD5_uUG3IRR9y81gYHDWRepgJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1m48NKmXKD5_uUG3IRR9y81gYHDWRepgJ" -O ./saved_models/model_59.ckpt.meta  && rm -rf /tmp/cookies.txt

python seq_to_seq_test.py $1 $2
