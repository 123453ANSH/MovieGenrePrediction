# MovieGenrePrediction
ML to predict movie genres based on posters and text
*Update documentation in README* 

Objective: Train a model that learns the features of a given movie poster in order to successfully predict the genre. 
If time allows, we would like to perform sentiment analysis on textual data extracted from movie trailers and reviews in order to augment our model.

Execution:
Data Analysis & Processing (April 19-21) 
Dataset: The Movies Dataset released by Kaggle 
Info: contains info 45,000 movies with data points including posters, backdrops, budget, revenue, release dates
movies_metadata.csv: contains posters and other visual elements that we will extract during preprocessing 
keywords.csv: contains movie plot keywords and critic reviews that may serve as textual data
Data preprocessing:
data augmentation to offset bias, will be done with one hot encoding movie genres
extract visual data and formatting each image into a 224*224*3 matrix to fit our model input size

Model Building (April 22-23) 
We plan to accomplish our goal using the ResNet34 architecture introduced in our main paper.
We will build a plain reference model as well as a custom architecture using KNN to compare the performance.

Plain 34 layer Baseline Model
easy to implement but falls short in performance, we will use this as reference model
ResNet34 (He etc. “Deep Residual Learning for Image Recognition”): 
higher accuracy as depth increases as a result of reducing the effect of the vanishing gradient problem
no significant computational cost in comparison to the plain net
KNN
simple implementation and lazy learners require no training time
as seen from previous papers, KNN may be less reliable than ResNet in the context of image feature classification
falls short in working with large datasets
extremely sensitive to outliers, need much more feature scaling/refinement on our dataset

Training & Debugging (April 24-28)
most time consuming portion of the project, we will dedicate a larger portion of our time to debug our model and augment it with additional features if time permits

Make & Practice Presentation (May 1-2nd)


