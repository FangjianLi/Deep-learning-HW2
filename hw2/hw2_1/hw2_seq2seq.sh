wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Mf-RWMAmS0jWEEExY6ZvFBJ-BufJg0PM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Mf-RWMAmS0jWEEExY6ZvFBJ-BufJg0PM" -O ./saved_models/model_69.ckpt  && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=159m7nISiNkV15DOGc4aAGp-pzmM8vEdI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=159m7nISiNkV15DOGc4aAGp-pzmM8vEdI" -O ./saved_models/model_69.ckpt.meta  && rm -rf /tmp/cookies.txt

python seq_to_seq_test.py $1 $2

