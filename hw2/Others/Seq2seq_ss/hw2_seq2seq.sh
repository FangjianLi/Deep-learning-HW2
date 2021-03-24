wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1u5geONy9E6-5RxfO5iPIiZpn5jMp2nIu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1u5geONy9E6-5RxfO5iPIiZpn5jMp2nIu" -O ./saved_models/model_59.ckpt  && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h9RfATUVp0nWrnyTaFhcKWeuURXQ-ouB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h9RfATUVp0nWrnyTaFhcKWeuURXQ-ouB" -O ./saved_models/model_59.ckpt.meta  && rm -rf /tmp/cookies.txt

python seq_to_seq_test.py $1 $2

