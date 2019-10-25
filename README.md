# MNIST ANN playground
A "I-am-bored-on-a-holiday-let's-play-with-neural-nets" project.

A neural network build from scratch, trained to work on MNIST handwritten digits.
It's slow, the implementation of the math is ugly, I was lazy documenting and writing it, but hey, I learned something.

## Description of the network
Input layer: consisting of 726*726 nodes.
One hidden layer of 32 nodes. Output activated by a sigmoid.
Classification is done with softmax.
Loss function is measured by cross entropy.
Batch traing update.

### TODO
Running the saved model yields different results. I guess some issues with an accumlated error in floating points.

## Running
Not that there is much to see, but:
```
# extract tar files
tar -xzvf ./input/testing.tar.gz -C ./input
tar -xzvf ./input/training.tar.gz -C ./input
# build
docker build -t cecemel/mnist-ann-playground .
# run
docker run -ti --rm -v "$PWD":/app -w /app cecemel/mnist-ann-playground python main.py
```
The main.py contains parameters to play with.

## Sources
[Neural networks: A visual introduction for beginnersb (Michael Taylor)](https://rkbookreviews.wordpress.com/2018/06/19/neural-networks-a-visual-introduction-for-beginners/)
[A nice video on how to calculate the gradient](https://www.youtube.com/watch?v=tIeHLnjs5U8)
[https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
[https://deepnotes.io/softmax-crossentropy](https://deepnotes.io/softmax-crossentropy)
