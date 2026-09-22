First of all , download dataset from web site http:s//physionet.org/content/challenge-2017/1.0.0/, only download the training2017.zip and REFERENCE-v3.csv, please put both files in the same folder. 

To run the code successfully, please download the software Anaconda3 from the official website(https://www.anaconda.com/download) and install it in advance.

1. After installation, open a Command Prompt (Windows) or Terminal (Linux), input the following command,
conda create -n CinC2017 python==3.8

2. Please activate the enrionment CinC2017 by running
conda activate CinC2017

3.Please install the pytorch framework and the below libraries by runing the following commands in sequence,
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c anaconda pillow scikit-learn pandas seaborn cython dbf
conda install -c conda-forge h5py timm einops yacs cvxpy nested_dict

4. Please navigate the directory where the prepare_dataset.py exists, to split the training data and validation data, 
please implement the following command,
python prepare_dataset.py

5. After the above command is executed,   the necessary files train.json and dev.json have been generated in config folder, please run the following command to train the model
python main.py

6. When trianing is finished, please run the following command, 
python test.py

7. if you don't spend time in training, a well-trained model is provided in the folder "./checkpoints", please run the following command, 
python statistic.py

8. if you want to see AUC for all classes and generate ROC curve, please run 
python ROC_AUC.py
