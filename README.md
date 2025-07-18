### Create Environment
1. Create Conda Environment
```
conda create --name oath python=3.10
conda activate oath
```

2. Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```
3. Install BasicSR
```
python setup.py develop --no_cuda_ext
```

### Pre-trained Model
- [Pre-trained Model for NTIRE 2025 Image Shadow Removal Challenge](https://mcmasteru365-my.sharepoint.com/:u:/g/personal/dongw22_mcmaster_ca/EUb0stsTQK9AhU7hIPpIaNIBrIb6kNp74SwKqV0Bju56DQ?e=ouCZV6).

### Our Submission on Test Sever
- [Our Test Output](https://mcmasteru365-my.sharepoint.com/:u:/g/personal/dongw22_mcmaster_ca/EZlSn_hUM_5DsZHiEyBB3RgBy19BP15WJ5b3TMUxC6U_xw?e=U9G4nj).

### Testing
Download above saved models and put it into the folder ./weights. To test the model, you need to specify the input image path (`args.input_dir`) and pre-trained model path (`args.weights`) in `./Enhancement/test_unpair.py`. Then run
```bash
python Enhancement/test_unpair.py 
```
You can check the output in `test-results-ntire25`.



### Contact
If you have any question, please feel free to contact us via dongw22@mcmaster.ca.

