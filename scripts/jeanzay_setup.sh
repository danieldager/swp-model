module purge
conda deactivate

module load pytorch-gpu/py3/2.5.0

pip install --user --upgrade --no-cache-dir g2p_en
pip install --user --upgrade --no-cache-dir levenshtein
pip install --user --upgrade --no-cache-dir morphemes
pip install --user --upgrade --no-cache-dir spacy
pip install --user --upgrade --no-cache-dir wordfreq

python -m spacy download en_core_web_lg

module purge
conda deactivate