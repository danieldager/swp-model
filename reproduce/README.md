## INSTRUCTIONS:
-------------

The following commands will allow you to reproduce the figures from the paper.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:danieldager/single-word-processing-model.git
   ```

2. Create a new environment:
   ```bash
   conda create -n nwr-model python=3.11
   conda activate nwr-model
   ```
   Or if you prefer using pyenv:
   ```bash
   pyenv virtualenv 3.11.0 nwr-model
   pyenv activate nwr-model
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en
   ```

### Reproducing Figures

To reproduce the figures, run the following commands in your terminal:

For example, to reproduce Figure 3:
```bash
python reproduce/figure3.py
```

![output](https://github.com/user-attachments/assets/0c161522-ef3f-4db3-be6b-cfc84d437bc3)


Generated figures will be saved in the `figures` directory.

You can also generate inside of the notebook `reproduce.ipynb`.
