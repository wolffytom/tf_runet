mkdir data
cd data
wget http://data.votchallenge.net/vot2016/vot2016.zip
wget http://cmp.felk.cvut.cz/~vojirtom/dataset/votseg/data/votseg_dataset.zip
mkdir vot2016
cd vot2016
mkdir images
unzip ../vot2016.zip -d images
mkdir groundtruths
unzip ../votseg_dataset.zip -d groundtruths
