#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from model import get_G
from model import SRGAN_d2 as get_D
from config import config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128

# ni = int(np.sqrt(batch_size))

# create folders to save result images and trained models
save_dir = "samples"
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = "models"
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]
    train_rgb_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.rgb_img_path, regx='.*.png', printable=False))#[0:20]

    # make sure pairs match, in the format of (FILENAME_gt.png, FILENAME_gi.png')
    assert all(file1.split('_')[0] == file2.split('_')[0] for (file1, file2) in zip(train_hr_img_list, train_rgb_img_list))
        # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
        # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
        # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_rgb_imgs = tl.vis.read_images(train_rgb_img_list, path=config.TRAIN.rgb_img_path, n_threads=32)
        # for im in train_hr_imgs:
        #     print(im.shape)
        # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
        # for im in valid_lr_imgs:
        #     print(im.shape)
        # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
        # for im in valid_hr_imgs:
        #     print(im.shape)

    # dataset API and augmentation
    def generator_train():
       for img, rgb in zip(train_hr_imgs, train_rgb_imgs):
            yield img, rgb

    def _map_fn_train(img_rgb_pair):
        img = img_rgb_pair[0][:,:,1:2] # use second channel in IR
        img=tf.expand_dims(img, 0)
        img.set_shape([1,240,320,1])
        rgb = img_rgb_pair[1][:,:,:3]
        rgb=tf.expand_dims(rgb, 0)
        # rgb.set_shape([None,240,320,3])
        # hr_patch = tf.image.random_crop(img, [384, 384, 3])
        img = img / (255. / 2.)
        rgb = rgb / (255. / 2.)
        img = img - 1.
        rgb = rgb - 1
        #hr_patch = tf.image.random_flip_left_right(hr_patch)
        #lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        lr_img = tf.image.resize(img, size=[60, 80])
        return lr_img[0], img[0], rgb[0]
        # return lr_img, img, rgb
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds

def train():
    G = get_G((batch_size, 60, 80, 1), (batch_size, 240, 320, 3))
    D = get_D((batch_size, 240, 320, 1))
    #VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    #VGG.train()

    train_ds = get_train_data()

    ## initialize learning (G)
    n_step_epoch = round(n_epoch_init // batch_size)
    for epoch in range(n_epoch_init):
        for step, (lr_imgs, imgs, rgbs) in enumerate(train_ds):
            if lr_imgs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_imgs = G([lr_imgs, rgbs])
                mse_loss = tl.cost.mean_squared_error(fake_imgs, imgs, is_mean=True)
            grad = tape.gradient(mse_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_imgs.numpy(), [3, 4], os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g_init.h5'))

    # adversarial learning (G, D)
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in range(n_epoch):
        for step, (lr_imgs, imgs, rgbs) in enumerate(train_ds):
            if lr_imgs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_imgs = G([lr_imgs, rgbs])
                logits_fake = D(fake_imgs)
                logits_real = D(imgs)
                # feature_fake = VGG((fake_imgs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                # feature_real = VGG((imgs+1)/2.)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
                # g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_imgs, imgs, is_mean=True)
                # vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                # g_loss = mse_loss + vgg_loss + g_gan_loss
                g_loss = (1e-2) * mse_loss + g_gan_loss
                grad = tape.gradient(g_loss, G.trainable_weights)
                g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
                grad = tape.gradient(d_loss, D.trainable_weights)
                d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

         # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_imgs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g_gan.h5'))
            D.save_weights(os.path.join(checkpoint_dir, 'd.h5'))

def evaluate(image_ids):
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    valid_rgb_img_list = sorted(tl.files.load_file_list(path=config.VALID.rgb_img_path, regx='.*.png', printable=False))


    assert all(file1.split('_')[0] == file2.split('_')[0] == file3.split('_')[0] for (file1, file2, file3) in zip(valid_lr_img_list, valid_lr_img_list, valid_rgb_img_list))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    valid_rgb_imgs = tl.vis.read_images(valid_rgb_img_list, path=config.VALID.rgb_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    def _map_fn_valid_ir(img):
        # img = img[:,:,1:2] # use second channel in IR
        img = img / (255. / 2.)
        img = img - 1.
        return img
    def _map_fn_valid_rgb(img):
        img = img / (255. / 2.)
        img = img - 1.
        img = img[:,:,:3]
        return img

    valid_lr_imgs = list(map(_map_fn_valid_ir, valid_lr_imgs))
    valid_hr_imgs = list(map(_map_fn_valid_ir, valid_hr_imgs))
    valid_rgb_imgs = list(map(_map_fn_valid_rgb, valid_rgb_imgs))

    ###========================== DEFINE MODEL ============================###
    G = get_G((1, None, None, 1), (1, None, None, 3))
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    for imid in image_ids:

        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        valid_rgb_img = valid_rgb_imgs[imid]

        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        # valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())


        valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
        valid_lr_img = valid_lr_img[np.newaxis,:,:,np.newaxis]
        valid_rgb_img = np.asarray(valid_rgb_img, dtype=np.float32)
        valid_rgb_img = valid_rgb_img[np.newaxis,:,:,:]
        size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

        out = G([valid_lr_img, valid_rgb_img]).numpy()

        print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
        tl.vis.save_image(valid_lr_img[0,:,:,0], os.path.join(save_dir, 'valid_lr.png'))
        tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))
        tl.vis.save_image(valid_rgb_img[0], os.path.join(save_dir, 'valid_rgb.png'))

        out_bicu = scipy.misc.imresize(valid_lr_img[0,:,:,0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))

        channels = out.shape[-1]
        psnr = -1
        ssim = -1
        for dim in range(channels):
            psnr = max(tf.image.psnr(
                out[0,:,:,dim:dim+1], valid_hr_img[:,:,np.newaxis], max_val=1
            ).numpy(), psnr)
            out_t = tf.convert_to_tensor(out[0,:,:,dim:dim+1])
            valid_hr_img_t = tf.cast(tf.convert_to_tensor(valid_hr_img[:,:,np.newaxis]), tf.float32)
            ssim = max(
                tf.image.ssim(out_t, valid_hr_img_t, max_val=1
                ).numpy(),ssim
            )

        print("PSNR for imid %d: %f" % (imid,psnr), flush=True)
        print("SSIM for imid %d: %f" % (imid,ssim), flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    parser.add_argument('--imid', type=str, default='ALL')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['imid'] = args.imid

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        imid = tl.global_flag['imid']
        if imid == 'ALL':
            all_ids = range(len(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False)))
            evaluate(all_ids)
        else:
            evaluate([int(imid)])

    else:
        raise Exception("Unknow --mode")
