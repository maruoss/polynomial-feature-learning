# Learning simple functions with 10 Neurons

Instructions on how to reproduce the gridplot of the simple functions in our paper "Polynomial Feature Learning".


## Getting started

**Open `main.py` and follow the steps there:**

1. To reproduce the gridplot for the simple functions of the paper set seed to 42.

2. Specify function variable out of [polynomialf, sinf, cosinef, expf, logf].
 - **polynomialf** for the (arbitrary) polynomial function $0.2x^{3} - 1.5x^{2} + 3.6x - 2.5$
 - **sinf** for $sin(x)$
 - **cosinef** for $cos(x)$
 - **expf** for $exp(x)$
 - **logf** for $log(x)$

3. Specify model class variable here:
 - **PolyModel** for our Polynomial NN
 - **StandardModel** for ReLU NN

4. Wait for training to end and use saved checkpoint "last.ckpt" later in Step 6.

5. Repeat with all other 9 combinations of functions and NN architectures.

6. When checkpoints for all functions and both network architectures are saved, specify their paths in the gridplotter.py file. The project folder should be recognized automatically, if not set the corresponding line at the start.

7. Run the `gridplotter.py` file to produce the plot.

8. Done!


## License
[MIT](https://choosealicense.com/licenses/mit/)