## INSTRUCTIONS:
-------------

The following commands will allow you to reproduce the figures from the paper.

### Environment Setup

1. Create a new environment:
   ```bash
   conda create -n nwr-model python=3.11
   conda activate nwr-model
   ```
   Or if you prefer using pyenv:
   ```bash
   pyenv virtualenv 3.11.0 nwr-model
   pyenv activate nwr-model
   ```

2. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```

### Reproducing Figures

To reproduce the figures, run the following commands in your terminal:

For example, to reproduce Figure 3:
```bash
python reproduce/figure3.py
```

Generated figures will be saved in the `figures` directory.

You can also generate inside of the notebook `reproduce.ipynb`.