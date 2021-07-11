# FEN_descriptor_chess

The goal of this project is to use Computer Vision and machine learning techniques to generate a Forsyth-Edwards Notation (FEN) descriptor from images. To explain briefly, FEN, for Forsyth–Edwards Notation, is a standard notation developed by Scottish for describing a particular board position of a chess game. So, it makes easy to translate any chess position into a single line of text. It facilitates the process of recreating positions using computers and allows players to share them and restart games from any point they desire. It replaces the need to send large PGN files.
<p align="center">
<img width="768" alt="image" src="https://user-images.githubusercontent.com/83417933/125186401-93234c80-e22a-11eb-8b2d-36f5dde0e2d5.png">
</p>

The following picture shows us briefly how the FEN descriptor works.
<p align="center">
<img width="350" alt="image" src="https://user-images.githubusercontent.com/83417933/125187637-ded8f480-e230-11eb-9268-5246c79b5476.gif">
</p>

## Data
The dataset comes from Kaggle (https://www.kaggle.com/koryakinp/chess-positions). It is composed of 100k images of a randomly generated chess positions of 5-15 pieces (2 kings and 3-13 pawns/pieces). Of course, i didn't use all this amount of images. Images (400x400 pixels) were generated using 28 styles of chess boards and 32 styles of chess pieces totaling 896 board/piece style combinations. Pieces were generated with the following probability distribution: 
- 30% for Pawn 
- 20% for Bishop
- 20% for Knight
- 20% for Rook
- 10% for Queen

Labels are in a filename in FEN format, but with dashes instead of slashes. This information will be very useful to label the images. 

## Methodology

The following schematic shows how I tackled the problem. I considered two approaches: one with a machine learning model and the other one with deep learning model. Then, it will be interesting to compare them. 
<p align="center">
<img width="746" alt="image" src="https://user-images.githubusercontent.com/83417933/125187877-054b5f80-e232-11eb-9264-2dfa75e666c6.png">
</p>

### Machine Learning approach
To use machine learning model, I need to compute features from images. The following schematic depicts how I processed. 
<p align="center">
 <img width="826" alt="image" src="https://user-images.githubusercontent.com/83417933/125191073-db4d6980-e240-11eb-843a-3dd63f633fac.png">
</p>

As you can see, there is two steps. I decided to process by two steps because as we can guess the majority of squares are empty, precisely 85%. So I can't expect an efficient classification. So I divided the problem in two steps. Let's explain these two steps:
- the first one is to classify whether a square is empty or not. First, I labelled whether the square is empty or not. Then, to feature it, I computed first the gradient and the HSV for Hue, Saturation and Value. From these two features, I computed several features:
  -	Mean of the gradient 
  -	Mean and standard deviation of Hue
  -	Mean and standard deviation of Saturation
  -	Mean and standard deviation of Value

I trained then my classifier. First, I trained a classic SVM and I obtained very good results.
<p align="center">
<img width="370" alt="image" src="https://user-images.githubusercontent.com/83417933/125191430-b2c66f00-e242-11eb-8691-116bff99714b.png">
</p>

Then, I trained a Knn to highlight which feature is important. As we can see, I can keep only two features and keep a high accuracy. This is in particular the mean of value and the mean of the gradient. 
<p align="center">
<img width="374" alt="image" src="https://user-images.githubusercontent.com/83417933/125191419-a510e980-e242-11eb-8ff7-457d5d53755e.png">
</p>

- the second step is to classify the type of pawn whether the square is not empty. To feature it, I computed the Histogram of Oriented Gradient (HOG). It is an efficient image descriptor to recognize shape is the histogram of oriented gradients. It is a technique that counts events of gradient orientation in a specific portion of an image or region of interest. As we can see in the following figure, after processing, we can clearly figure out the type of pawn.
<p align="center">
 <img width="150" alt="image" src="https://user-images.githubusercontent.com/83417933/125190276-97f0fc00-e23c-11eb-9c47-3115a00deee5.png">
</p>

After computing the feature, I trained a classifier. I trained a SVM and Knn like before. You can see the following picture the confusion matrix of the Knn classifier whose I got the best results.
<p align="center">
<img width="371" alt="image" src="https://user-images.githubusercontent.com/83417933/125191605-9e36a680-e243-11eb-83c8-9b5686f4e615.png">
</p>

We can notice that the pawn 'Q' and 'q' have the "worst" accuracy. I could guess it due to the distribution of pawn:
<p align="center">
<img width="297" alt="image" src="https://user-images.githubusercontent.com/83417933/125191761-5b290300-e244-11eb-8161-de1307d8a8e5.png">
</p>

After these two steps, I tested my models to evaluate it. I obtained only 65%. It could be suprising due to the high efficiency of classifiers but it is enough to misclassify an square to have all the FEN descriptor of the chess boards false. But I think I can improve it easily by training classifier with more data in particular the second one. 

### Deep Learning approach



