'''
WganGp Model

Updated on 2017.07.31
Author : Yeonwoo Jeong
'''
from ops import mnist_for_gan, optimizer, clip, get_shape, softmax_cross_entropy, sigmoid_cross_entropy
from config import WganGpConfig , SAVE_DIR, PICTURE_DIR
from utils import show_gray_image_3d, make_gif, create_dir
from nets import GenConv, DisConv
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import glob
import os

logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sample_z(z_size, z_dim):
    return np.random.uniform(low=-1, high=1, size= [z_size, z_dim])

class WganGp(WganGpConfig):
    def __init__(self):
        WganGpConfig.__init__(self)
        logger.info("Building model starts...")
        tf.reset_default_graph()
        self.generator = GenConv(name ='g_conv', batch_size=self.batch_size)
        self.discriminator = DisConv(name='d_conv')
        self.dataset = mnist_for_gan()
        
        self.X = tf.placeholder(tf.float32, shape = [self.batch_size, self.x_size, self.x_size, self.x_channel])
        self.Z = tf.placeholder(tf.float32, shape = [self.batch_size, self.z_dim])
        
        self.G_sample = self.generator(self.Z)
        self.D_real = self.discriminator(self.X)
        self.D_fake = self.discriminator(self.G_sample, reuse = True)

        # x_hat = epsilon*x_real + (1-epsilon)*x_gen
        self.epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)# epsilon : sample from uniform [0,1]
        self.linear_ip = self.epsilon*self.X + (1-self.epsilon)*self.G_sample
        self.D_ip = self.discriminator(self.linear_ip, reuse=True)
        self.gradient = tf.gradients(self.D_ip, [self.linear_ip])[0]
        self.gradient_penalty = tf.reduce_mean(tf.square(tf.norm(self.gradient, axis=1) - 1.))

        self.G_loss = tf.reduce_mean(self.D_real)-tf.reduce_mean(self.D_fake)   
        self.D_loss = -self.G_loss+self.lamb*self.gradient_penalty  
        
        '''
        Normal gan with KL
		self.D_loss = tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_real, labels=tf.ones_like(self.D_real))) + tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(sigmoid_cross_entropy(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
		'''

        self.generator.print_vars()
        self.discriminator.print_vars()

        self.D_optimizer = optimizer(self.D_loss, self.discriminator.vars)
        self.G_optimizer = optimizer(self.G_loss, self.generator.vars)
        '''
        deprecate weight clipping stead use gradient penalty stands for gp

        self.clip_b = tf.Variable(self.clip_b, trainable=False, name="clipper")
        with tf.control_dependencies([self.D_optimizer]):
            self.D_optimizer_wrapped = [tf.assign(var, clip(var, -self.clip_b, self.clip_b)) for var in self.discriminator.vars]
        '''     


        # Fixed sample
        self.z_sample_fix = sample_z(self.batch_size, self.z_dim)
        logger.info("Building model done.")
        self.sess = tf.Session()
        
    def initialize(self):
        """Initialize all variables in graph"""
        logger.info("Initializing model parameters")
        self.sess.run(tf.global_variables_initializer())

    def restore(self):
        """Restore all variables in graph"""
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")     
    
    def sample_data(self, fix=False):
        """sampling for data
        Return:
            X_sample, z_sample, c_sample
        """
        X_sample = self.dataset(self.batch_size)
        z_sample = sample_z(self.batch_size, self.z_dim)

        if fix:
            return X_sample, self.z_sample_fix
        else:
            return X_sample, z_sample

    def train(self, train_epochs):
        count = 0
        for epoch in tqdm(range(train_epochs), ascii = True, desc = "batch"):
            if epoch < 25:
                d_iter = 100
            else:
                d_iter = 5
            for _ in range(d_iter):
                X_sample, z_sample = self.sample_data()
                self.sess.run(self.D_optimizer, feed_dict = {self.X : X_sample, self.Z : z_sample})
            
            for _ in range(1):
                X_sample, z_sample = self.sample_data()
                self.sess.run(self.G_optimizer, feed_dict = {self.X : X_sample, self.Z : z_sample})
                
            if epoch % self.log_every == self.log_every-1:

                X_sample, z_sample= self.sample_data(fix=True)
                D_loss = self.sess.run(self.D_loss, feed_dict = {self.X : X_sample, self.Z : z_sample})
                G_loss = self.sess.run(self.G_loss, feed_dict = {self.X : X_sample, self.Z : z_sample})
                logger.info("Epoch({}/{}) D_loss : {}, G_loss : {}".format(epoch+1, train_epochs, D_loss, G_loss))

                # Save Picture
                count+=1
                gray_3d = self.sess.run(self.G_sample, feed_dict = {self.Z : z_sample}) # self.batch_size x 28 x 28 x 1
                gray_3d = np.squeeze(gray_3d)#self.batch_size x 28 x 28
            	# Store generated image on PICTURE_DIR
                fig = show_gray_image_3d(gray_3d, col=10, figsize = (50, 50), dataformat = 'CHW')
                fig.savefig(PICTURE_DIR+"%s.png"%(str(count).zfill(3)))
                plt.close(fig)

                # Save model
                saver=tf.train.Saver(max_to_keep = 10)
                saver.save(self.sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch+1)
                logger.info("Model save in %s"%SAVE_DIR)


if __name__=='__main__':
    create_dir(SAVE_DIR)
    create_dir(PICTURE_DIR)
    wgangp = WganGp()
    wgangp.initialize()
    wgangp.train(100000)

    images_path = glob.glob(os.path.join(PICTURE_DIR, '*.png'))
    gif_path = os.path.join(PICTURE_DIR, 'Movie.gif')
    make_gif(sorted(images_path), gif_path)