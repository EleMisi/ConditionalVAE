## Instructions

-----------------------------------

1. Download the repository
2. Upload the repository on Google Drive
3. Mount the Drive on a new notebook
4. Configure Kaggle API
5. Run in a cell `%cd /content/drive/My\ Drive/ConditionalVAE` (to move to the repository folder)
5. Run in a cell `!python3 dataloader.py` (to download the dataset)
6. Run in a cell `!python3 train.py -n 64 -nn Conv -l 0.001 -b 2 -bs 32 -e 10 -p 1 -td 0.8 -d 0.5 -c 10` (to start training the model - read below for further information about the train configuration)
7. Run in a cell `!python3 plot.py` (to plot the generated images - read below for futher information about the plot configuraion)


## Train and plot configurations

### Train positional arguments
* -a -> alpha parameter
* -b -> beta parameter
* -bs -> batch size 
* -c -> clipping gradient parameter
* -d -> dropout parameter
* -e -> number of training epochs
* -l -> learning rate
* -n -> latent space dimension
* -nn -> neural network ['Dense' : dense hidden layers, 'Conv' : convolutional layers]
* -p -> plot parameter (0 means no plot, 1 means plot)
* -td -> training set dimension wrt the whole dataset


### Plot positional arguments
* -m -> plotting mode ['reconstr' : plot only reconstructed images, 'gen' : to plot generated images] 
* -n -> latent space dimension
* -nn -> neural network ['Dense' : dense hidden layers, 'Conv' : convolutional layers]
* -t -> generate images with specific attributes or not [0 : generate images with attributes taken from the test set, 1 : generate images with user specified attributes]

The images are saved in a folder named result



