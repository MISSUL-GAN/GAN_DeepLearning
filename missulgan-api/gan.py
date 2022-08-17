import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html, util
from PIL import Image
import shutil

def convert_gan(convert_tag, origin_img, token):
    opt = TestOptions().parse(convert_tag)  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images_cnn are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images_cnn are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    createDataroot(token, origin_img)
    opt.dataroot = token
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images_cnn.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images_cnn to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                    use_wandb=opt.use_wandb)
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data)
            image_pil = Image.fromarray(im)
            # image_pil.save(image_path)
        deleteDataroot("testDir")
        return image_pil

def createDataroot(data_path, data):
    os.mkdir(data_path)
    image = Image.open(data)
    image.save(data_path + "/data4.jpg")

def deleteDataroot(data_path):
    shutil.rmtree(data_path)
"""
if __name__ == '__main__':
    opt = TestOptions().parse("style_ukiyoe_pretrained")  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images_cnn are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images_cnn are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    createDataroot("testDir", "images_gan/origin2.jpg")
    opt.dataroot = "testDir"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images_cnn.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images_cnn to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    deleteDataroot("testDir")
    webpage.save()  # save the HTML
"""