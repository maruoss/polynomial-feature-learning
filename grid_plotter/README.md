# Learning simple functions with 10 Neurons

Instructions on how to reproduce the gridplot of the toy examples in our paper "Polynomial Feature Learning".


## Getting started

**1) Requirements**

Preferably, create your own conda environment before following the step below:

```bash
conda env create -f environment.yml
```

**2) Steps**

Open `main.py` and follow the steps there:

1. To reproduce the toy plots of the paper set seed to 42.

2. Choose function out of [polynomialf, sinf, cosinef, expf, logf].

3. Specify model class here:
 - "PolyModel" for our Polynomial NN
 - "StandardModel" for ReLU NN

4. Wait for training to end and use saved checkpoint "last.ckpt" later in Step 6.

5. Repeat with all other 9 combinations of functions and NN architectures.

6. When checkpoints for all functions and both network architectures are saved, specify their paths in the gridplotter.py file. The project folder should be recognized automatically, if not set the corresponding line at the start.

7. Run the `gridplotter.py` file to produce the plot.

8. Done!


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)