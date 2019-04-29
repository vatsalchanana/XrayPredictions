# Deep learning for Chest X-ray diagnosis
In the US, over a 150 million chest X-rays are done yearly by radiologists to help doctors with diagnosis. Chest X-rays are currently the best available method for diagnosing diseases like pneumonia. Automated chest radiograph interpretation at the level of practicing radiologists could provide substantial benefit in many medical settings. In this blog, we outline how we tried to automate this crucial aspect of modern medicine using deep learning approaches. 

### Dataset description

The dataset of Chest X-rays that we will be using for the project is called CheXpert collated by the Stanford ML group. We use the lowerr resolution dataset. The dataset can be found here : https://stanfordmlgroup.github.io/competitions/chexpert/

### Running Models:
To run the models, change the path for the files in the dataloaders.

You can run the following models in the following files:

Logistic Regression 4 class: 4ClassLR.ipynb
Logistic Regression 14 class: LRWithReplacement.ipynb

FFNN with 4 classes: FFN.ipynb

CNN with 4 classes: CNN_4classes.ipynb
CNN with 14 classes: CNN-14C.ipynb

DenseNet121: DenseNet_14classes_SingleModel.ipynb

ResNet : resnet_train.py, torch_train_resnet.py

Combined Model: MultipleModels14C.ipynb

Testing models on test set: TestModels (1).ipynb

CAM: Cam.py

Guided backprop: Guided Backprop.ipynb

Hypothesis testing: TestingHypothesis.ipynb


