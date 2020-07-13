import keras as k
from keras import Input, Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Conv2D, Add, BatchNormalization, Concatenate
from keras.applications import VGG19
from os import path
from model.layer_utils import *
from code.utils import load_data, guided_filter
from code.losses import RTV_loss

import datetime
from glob import glob
import numpy as np
import datetime
import cv2
import os
import click
from skimage.measure import compare_psnr, compare_ssim

# Base path
base_path = '/Users/albert/con_lab/alyssa/remodel'

# data
Heavy_rain_path = glob('/Users/albert/con_lab/alyssa/Heavy_rain/*')
Total_rain_path = glob('/Users/albert/con_lab/alyssa/Dataset_Rain/*')

# weight file
g_weight_path = path.join(base_path,'/Users/albert/con_lab/alyssa/remodel/weights/new_weights/78/generator_50000.h5')
d_weight_path = path.join(base_path,'/Users/albert/con_lab/alyssa/remodel/weights/new_weights/78/discriminator_50000.h5')

# model file 
# make code clear    
g_model_path = path.join(base_path, 'model/g_model.h5')
d_model_path = path.join(base_path, 'model/d_model.h5')
save_path = path.join(base_path, 'new_out')
weight_save = path.join(base_path, 'weight/new_weights')
save_num = 3
save_epoch = 50

is_load_weight = False
continue_train = False
batch_size = 4
# continue_weight_name = 'con_out_0_3916.h5'

image_size = (512, 512, 3)
g_trainable = True


def save_all_weights(final_g, final_d, epoch_number):
    now = datetime.datetime.now()
    save_dir = path.join(weight_save, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    final_g.save_weights(
        os.path.join(save_dir, 'generator_{}.h5'.format(
            epoch_number)), True)
    final_d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


class detrain():
    def __init__(self):
        self.generator = self.bulid_generator()
        self.d = self.bulid_discriminator()

        self.vgg = self.bulid_VGG()
        self.vgg.trainable = False

        I_details = Input(image_size)
        I_bases = Input(image_size)

        f_I_details = self.generator([I_details, I_bases])
        f_features = self.vgg(f_I_details)
        disc = self.d(f_I_details)

        self.d.trainable = True
        self.d.compile(optimizer='adam', loss='binary_crossentropy')
        self.d.trainable = False

        self.combained_with_vgg = Model([I_details, I_bases], [disc, f_features])
        self.combained_with_vgg.compile('adam', loss=['binary_crossentropy', 'mse'], loss_weights=[1, 100])

        gen_adam = Adam(lr=0.0001)
        self.generator.compile(gen_adam, 'mse')
        self.output_true_batch = np.ones((batch_size, 1))
        self.output_false_batch = np.zeros((batch_size, 1))

        self.d.trainable = True
        
    def bulid_generator(self):

        generator = load_model(
            g_model_path,
            custom_objects={
                'ReflectionPadding2D': ReflectionPadding2D
            })
        if is_load_weight:
            generator.load_weights(g_weight_path)
        
        I_bases = Input(image_size)
        I_details = Input(image_size)

        fake_details = generator(I_details)
        fake_x = Add()([fake_details, I_bases])

        return Model([I_details, I_bases], [fake_x])

    def bulid_discriminator(self):
        discriminator = load_model(d_model_path)
        # discriminator.load_weights(d_weight_path)
        return discriminator

    def bulid_VGG(self):
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape=image_size)
        img_features = vgg(img)
        return Model(img, img_features)

    def train_combained(self, image_detials, image_bases, real_image, epcho, batch_size):

        real_features = self.vgg.predict(real_image)

        totoal_loss = []
        d_losses = []
        generated = self.generator.predict([image_detials, image_bases])

        for i in range(epcho):
            d_loss_real = self.d.train_on_batch(real_image, self.output_true_batch)
            d_loss_fake = self.d.train_on_batch(generated, self.output_false_batch)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            d_losses.append(d_loss)
            
        com_loss = self.combained_with_vgg.train_on_batch([image_detials, image_bases], [self.output_true_batch, real_features])
        totoal_loss.append(com_loss)
        
        d_total_loss = np.mean(d_losses)
        totoal_loss = np.mean(totoal_loss)
        
        return totoal_loss, d_total_loss
    
    def train_generator(self, image_detials, image_bases, real_image, epcho):

        totoal_loss = 0
        for i in range(epcho):
            com_loss = self.generator.train_on_batch([image_detials, image_bases], real_image)
            totoal_loss += com_loss
        
        totoal_loss = np.mean(totoal_loss)
        
        return totoal_loss
        

def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
    net = detrain()
    #net.generator.load_weights('/Users/albert/con_lab/alyssa/remodel/weight/new_weights/77/generator_43000.h5')
    #net.d.load_weights('/Users/albert/con_lab/alyssa/remodel/weight/new_weights/77/discriminator_43000.h5')
    

    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch, epoch_num))

        y_pre, x_pre = load_data(Heavy_rain_path, batch_size)

        x_base = guided_filter(x_pre, batch_size, 512, 3)

        rain_detail = x_pre - x_base

        details_loss, dis_loss = net.train_combained(rain_detail, x_base, y_pre, critic_updates, batch_size)
        print('batch {} dis_loss : {}'.format(epoch, dis_loss))
        print('batch {} detials_loss : {}'.format(epoch, details_loss))

        # 计算mse损失
        y_pre, x_pre = load_data(Total_rain_path, batch_size)
        x_base = guided_filter(x_pre, batch_size, 512, 3)
        rain_detail = x_pre - x_base

        color_loss = net.train_generator(rain_detail, x_base, y_pre, 1)
        print('batch {} color_loss : {}'.format(epoch, color_loss))

        if epoch % save_epoch == 0:     
            img_ge = net.generator.predict([rain_detail[0:save_num], x_base[0:save_num]])

            for i in range(save_num):
                img_ge1 = np.array(img_ge[i, :, :, :], dtype=np.float32)
                img_fu = np.array(y_pre[i, :, :, :], dtype=np.float32)
                img_bl = np.array(x_pre[i, :, :, :], dtype=np.float32)
               
                img_fu = (img_fu + 1) * 127.5
                img_bl = (img_bl + 1) * 127.5
                img_ge1 = (img_ge1 + 1) * 127.5

                ori_nn_psnr = compare_psnr(img_ge1, img_fu, 255)
                ori_nn_ssim = compare_ssim(img_ge1, img_fu, multichannel=True, data_range=255)
                
                cv2.putText(img_ge1, f'ssim:{ori_nn_ssim}', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
                cv2.putText(img_ge1, f'psnr:{ori_nn_psnr}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
                output = np.concatenate(
                    (img_ge1, img_fu, img_bl), axis=1)
                img_name = str(epoch) + '_' + str(i) + '.jpg'
                cv2.imwrite(path.join(save_path, img_name), output)

            with open(path.join(base_path, 'log.log'), 'a') as f:
                f.write('{} - {} - {}\n'.format(epoch, np.mean(details_loss),
                                                np.mean(color_loss)))

        if (epoch % 100 == 0):
            save_all_weights(net.generator, net.d,  epoch)


@click.command()
@click.option(
    '--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=4, help='Size of batch')
@click.option(
    '--epoch_num', default=70001, help='Number of epochs for training')
@click.option(
    '--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, epoch_num,
                                  critic_updates)


if __name__ == '__main__':
    train_command()
