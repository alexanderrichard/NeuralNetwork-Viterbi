# NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
Code for the paper NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning

### Prepraration:

* download the data from https://uni-bonn.sciebo.de/s/vVexqxzKFc6lYJx
* extract it so that you have the `data` folder in the same directory as `train.py`
* create a  `results` directory in the same directory where you also find `train.py`: `mkdir results`

Requirements: Python2.7 with the libraries numpy and pytorch

### Training:

Run `./train.py`

### Inference:

Run `./infernece.py`
Note: adjust the variable `n_threads` in `inference.py` to your needs.

### Evaluation:

In the inference step, recognition files are written to the `results` directory. The frame-level ground truth is available in `data/groundTruth`. Run `./eval.py --recog_dir=results --ground_truth_dir=data/groundTruth` to evaluate the frame accuracy of the trained model.

### Remarks:

We provide a python/pytorch implementation for easy usage. In the paper, we used an internal C++ implementation, so results can be slightly different. Running the provided setup on split1 of Breakfast should lead to roughly 42% frame accuracy.

If you use the code, please cite

    A. Richard, H. Kuehne, A. Iqbal, J. Gall:
    NeuralNetwork-Viterbi: A Framework for Weakly Supervised Video Learning
    in IEEE Int. Conf. on Computer Vision and Pattern Recognition, 2018
