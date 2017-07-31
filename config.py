'''
Basic operation not based on tensorflow

Updated on 2017.07.31
Author : Yeonwoo Jeong
'''

#==================================================PATH===================================================#

SAVE_DIR = './save/'
MNIST_PATH =  "../MNIST_data"
PICTURE_DIR = './asset/'

#===========================================InfoGAN configuraion===========================================#
class WganGpConfig(object):
    def __init__(self):
        self.x_channel = 1
        self.x_size = 28
        self.x_dim = 784
        self.lamb = 10
        self.z_dim = 100

        self.batch_size = 100
        self.log_every = 1000

        self.clip_b = 0.0001# clip bounday variable to be clipped to be in [-self.clip_b, self.clip_b]
