# spatial_label_smoothing
Spatial Label Smoothing for PyTorch

In order to run this experiment you need:
- Python >= 3.7
- PyTorch >= 1.6
- Torchmetrics
- GDAL >= 3.2
- Buteo > 0.9.50

To install GDAL and Buteo, you can use the following commands:
```
conda install -c conda-forge gdal=3.6.2
pip install buteo
```

The main script is `train.py`.
To generate the visualizations and results, you can use the notebook `visualizations.ipynb`.