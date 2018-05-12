# MusicAI
Automated Music Assistant for Amateur Pianists

This project was created with the intention of assisting amateur pianists by accompanying the melody the play (with the right hand) with accompanying chords and arpeggios (played on the left). This serves the dual purpose of assisting students who are capable of playing with only one hand in their stage of learning with accompanying background that both makes them sound better as well as encourages them to continue learning.

## Results

In order to achieve the proposed goal, we implemented several machine learning models that predict chords from the melody being played. These include sequence models such as HMM and a KNN+Simple Markov Model combination, classsifiers such as MLP, Logistic Regression, and SVM, and deep models such as LSTM and CNN.

Our best results are currently using MLP with lbfgs solver at 48.8 percent.

## Data
We have created our own dataset of 44 songs for the purpose of this project, including 24 contemporary pop songs, 15 nursery rhymes, and 5 improvisational pieces. These were recorded in the MIDI format and can be found here - https://www.dropbox.com/s/40fvkhb678rqzt7/chord_prediction_dataset.rar?dl=0
