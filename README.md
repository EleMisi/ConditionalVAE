# How to run on Colab:

1. Download the repository
2. Load on you Google Drive
3. Open Colab
4. Mount Drive (Files -> Mount Drive)
5. Run `%cd /content/drive/My Drive/ConditionalVAE`
6. Run `!python3 train.py -n 20 -c 1 -l 0.001 -e 10` (-e stands for number of epochs)
7. To plot some temporaneal results: `!python3 plot.py -n 20`
