## Instructions

-----------------------------------

1. Download the repository
2. Upload the repository on Google Drive
3. Mount the Drive on a new notebook
4. Configure Kaggle API
5. run in a cell `%cd /content/drive/My\ Drive/ConditionalVAE` (to move to the repository folder)
5. run in a cell `!python3 dataloader.py` (to download the dataset)
6. run in a cell `!python3 train.py -n 20 -l 0.001 -b 32 -e 10 -p 0 -d 0.8` (to start training the model - read below for further information about the train configuration)
7. run in a cell `!python3 plot.py -m gen -t 0` (to plot the generated images - read below for futher information about the plot configuraion)



