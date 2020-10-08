# gait_recognition

Implementation of a continuous gait recognition in the verification scenario.

The first task is to load the dataset. 
We published a gait dataset called RecodGait_v1: https://figshare.com/articles/RecodGait_v1/4975028

Afterwards, the next task is apply a feature space to the data. 
In the verification scenario, we are dealing with unbalanced data thus, we perform a down-sampling in the training data.
We adopted the min-max normalization for the features and then, a grid search is applied in order to find the best parameters for a desired classifier. 

At the end, the test set is evaluated by the trained classifier and then, the results are presented considering the usual metrics in biometrics (recall, precision, etc.). 

This implementation was product from a master's thesis and all details about the techniques can be found on (Portuguese):
Geise Santos. Continuous gait authentication techniques for mobile devices. Campinas: Institute of Computing, University of Campinas, 2017. Master's thesis in Computer Science.
http://repositorio.unicamp.br/bitstream/REPOSIP/322672/1/Santos_GeiseKellyDaSilva_M.pdf
