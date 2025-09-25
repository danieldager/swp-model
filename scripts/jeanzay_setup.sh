module purge
conda deactivate

git submodule init
git submodule update

cp -r ../stimuli/ $WORK

module load pytorch-gpu/py3/2.5.0

# pip install --user --upgrade --no-cache-dir g2p_en
# pip install --user --upgrade --no-cache-dir levenshtein
# pip install --user --upgrade --no-cache-dir morphemes
# pip install --user --upgrade --no-cache-dir spacy
# pip install --user --upgrade --no-cache-dir wordfreq

pip install --upgrade --no-cache-dir g2p_en
pip install --upgrade --no-cache-dir levenshtein
pip install --upgrade --no-cache-dir morphemes
pip install --upgrade --no-cache-dir spacy
pip install --upgrade --no-cache-dir tensordict
pip install --upgrade --no-cache-dir wordfreq

if ! python -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
    python -m spacy download en_core_web_lg
else
    echo "SpaCy model already installed, skipping download"
fi

python -m nltk.downloader "averaged_perceptron_tagger" "cmudict"

module purge
conda deactivate

mkdir -p $WORK/weights/cornet
wget https://s3.amazonaws.com/cornet-models/cornet_z-5c427c9c.pth -P $WORK/weights/cornet
wget https://s3.amazonaws.com/cornet-models/cornet_rt-933c001c.pth -P $WORK/weights/cornet
wget https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth -P $WORK/weights/cornet
wget https://s3.amazonaws.com/cornet-models/cornet_r-5930a990.pth -P $WORK/weights/cornet