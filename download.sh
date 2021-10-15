pip install spacy==2.1.8 transformers datasets langid faker nltk sentencepiece fsspec tqdm
python -m nltk.downloader punkt stopwords  wordnet
python -m spacy download en_core_web_lg
gdown https://drive.google.com/u/0/uc?id=1-9Wyu7ImEX8W21P701LeyTeIEFWpvixu&export=download
gdown https://drive.google.com/u/0/uc?id=1-3066SYYuBE_d-zGkqpEX_reTlmYDo9Z&export=download
python create_pii_dataset/load.py
