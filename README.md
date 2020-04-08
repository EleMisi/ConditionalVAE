## Instructions

-----------------------------------

1. Download the repository
2. Upload the repository on Google Drive
3. Mount the Drive on a new notebook
4. Configure Kaggle API
5. Run in a cell `%cd /content/drive/My\ Drive/ConditionalVAE` (to move to the repository folder)
5. Run in a cell `!python3 dataloader.py` (to download the dataset)
6. Run in a cell `!python3 train.py -n 20 -l 0.001 -b 32 -e 10 -p 0 -td 0.8` (to start training the model - read below for further information about the train configuration)
7. Run in a cell `!python3 plot.py` (to plot the generated images - read below for futher information about the plot configuraion)


## Train and plot configurations

### Train positional arguments
* -n -> latent space dimension
* -l -> learning rate
* -b -> batch size (keep it small, < 100)
* -e -> number of training epochs
* -d -> dropout parameter (0 means no dropout)
* -td -> training set dimension wrt the whole dataset
* -p -> plot parameter (0 means no plot, 1 means plot)

### Plot positional arguments
* -m -> plotting mode (reconstr to plot only reconstructed images, gen to plot only generated images; deafult value is None and both the plot types are performed)
* -t -> generate specific attributes or random (0 to generate images with random attributes, 1 to generate images with specific attributes)

The images are saved in a folder named result



