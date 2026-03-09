# DMM-MEN Notebook

Files:

- `mnist_dmm_men_colab.ipynb`
- `mnist_mlp.py`
- `dmm_men_core.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

Install:

```bash
python -m pip install -r requirements.txt
jupyter notebook mnist_dmm_men_colab.ipynb
```

Run:

- Local:

```bash
jupyter notebook mnist_dmm_men_colab.ipynb
```

- Colab:
  - open `mnist_dmm_men_colab.ipynb` in Colab
  - run the first cell
  - the notebook clones `Vehils-Vinals/DL_DMM_MEM`, installs `requirements.txt`, and adds the repo to `sys.path`

Notebook content:

- `MNIST`: global and local comparison between DMM-MEN and LIME
- `ImageNet / VGG16`: local comparison only
