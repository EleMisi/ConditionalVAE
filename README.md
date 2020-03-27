## Work in Porgress

-----------------------------------

# How to run on Colab:

1. Download the repository
2. Load on your Google Drive
3. Load on your Google Drive, inside the ConditionalVAE folder, a folder named "ProvvisorialData" with the files arr_no_Eyeglasses.pickle and img_no_Eyeglasses.pickle 
4. Open Colab
5. Mount Drive (Files -> Mount Drive)
6. Run `%cd /content/drive/My Drive/ConditionalVAE`
7. Run `!python3 train.py -n 20 -c 1 -l 0.001 -e 100 -b 1000` (-e stands for number of epochs, _b stands for the batch size, -l is the leraning rate, -n is the latent space dimension)
8. NOT AVAILABLE! To plot some temporaneal results: `!python3 plot.py -n 20`
