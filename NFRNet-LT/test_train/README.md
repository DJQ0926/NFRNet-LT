NFRNet-LT:A new neural network for solving non-functional requirement classification based on the loss function of the exponential moving average (EMA)algorithm
==================================


#### Run code as follows:
* 1.Download bert-base-uncased(e.g. hugging face) into "bert-base" and configure the path to bert in "config.py".<br>

* 2.Add the dataset to "/data/input"and configure it in the "conifg.py" path.<br>

* 3.Run "train.py" and "test.py".<br>

requrements:
Package              Version
-------------------- -----------
matplotlib           3.5.1
numpy                1.21.6
pandas               1.2.4
scikit-learn         1.0.2
spacy                2.2.0
torch                1.7.1+cu110
torchaudio           0.7.2
torchtext            0.9.0
torchvision          0.8.2+cu110
transformers         4.16.2
