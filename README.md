# Cox regression experiments

Code for the five-method benchmark and approximation diagnostics. Requires Python 3.10.

```sh
python -m pip install -r requirements.txt
python run.py --dataset nwtco --download --output results
python diagnostics.py --download --output diagnostics
```

Run these commands from this folder. Use `--dataset all` for all five datasets.
Parameters are in `settings.py`. Real data are downloaded; synthetic data are generated.
Results are saved as NPZ files, readable with `numpy.load`. Use a new output folder for each run.
The diagnostics command reproduces Figure 2 and Table 8 using reference coefficients in `settings.py`.
