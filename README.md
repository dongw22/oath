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


### Pre-trained Model
- [Pre-trained Model for NTIRE 2025 Image Shadow Removal Challenge](https://mcmasteru365-my.sharepoint.com/:u:/r/personal/dongw22_mcmaster_ca/Documents/2025NTIRE_shadow_removal/net_g_9600.pth?csf=1&web=1&e=JwsKhJ).

### Our Submission on Test Sever
- [Our Test Output](https://mcmasteru365-my.sharepoint.com/:u:/r/personal/dongw22_mcmaster_ca/Documents/25NTIRE_reflection_removal/shadow_test_result.zip?csf=1&web=1&e=pWtEb5).

### Testing
Download above saved models and put it into the folder ./weights. To test the model, you need to specify the input image path (`args.input_dir`) and pre-trained model path (`args.weights`) in `./Enhancement/test_unpair.py`. Then run
```bash
python Enhancementy/test_unpair.py 
```
You can check the output in `test-results-ntire25`.



### Contact
If you have any question, please feel free to contact us via dongw22@mcmaster.ca.

