#!/bin/bash

mkdir models

cd ./models
git clone https://github.com/TGoldsack1/LENS.git
pip install LENS/lens/.

curl "https://drive.usercontent.google.com/download?id=179cuRZdJZEtEObovVf_KPhNFWnMF8pkN&confirm=xxx" -o LENS-checkpoint.zip

unzip LENS-checkpoint.zip -d ./LENS
rm LENS-checkpoint.zip
echo "strict: False" >> ./LENS/LENS/hparams.yaml 

git clone https://github.com/yuh-zha/AlignScore.git
pip install AlignScore/.
python -m spacy download en_core_web_sm
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-base.ckpt -P ./AlignScore

cd ../