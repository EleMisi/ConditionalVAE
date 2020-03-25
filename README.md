# How to run on Colab:

1. Download the repository
2. Load on your Google Drive
3. Load on your Google Drive, inside the ConditionalVAE folder, a folder named "ProvvisorialData" with the files arr_no_Eyeglasses.pickle and img_no_Eyeglasses.pickle 
3. Open Colab
4. Mount Drive (Files -> Mount Drive)
5. Run `%cd /content/drive/My Drive/ConditionalVAE`
6. Run `!python3 train.py -n 20 -c 1 -l 0.001 -e 10 -b 1000` (-e stands for number of epochs, _b stands for the batch size, -l is the leraning rate, -n is the latent space dimension)
------Not Available--------
7. To plot some temporaneal results: `!python3 plot.py -n 20`
---------------------------