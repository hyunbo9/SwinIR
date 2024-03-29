import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from data.dataset_multi_sr import RandomResizeCollater

from hyunbo import imresize

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''

def main(json_path='options/swinir/train_swinir_sr_hyunbo_default.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                random_resize_collate_fn = RandomResizeCollater(patch_size=opt['datasets']['train']['H_size'], sf_list= opt['scale'])
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler,
                                          collate_fn=random_resize_collate_fn)
            else:
                random_resize_collate_fn = RandomResizeCollater(patch_size=opt['datasets']['train']['H_size'], sf_list= opt['scale'])
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True,
                                          collate_fn=random_resize_collate_fn)

        elif phase == 'test':
            set5_path = dataset_opt['test_set']['set5']
            set14_path = dataset_opt['test_set']['set14']

            dataset_opt['dataroot_H'] = set5_path 
            set5_test_set = define_Dataset(dataset_opt)
            set5_test_loader = DataLoader(set5_test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True,)

            dataset_opt['dataroot_H'] = set14_path 
            set14_test_set = define_Dataset(dataset_opt)
            set14_test_loader = DataLoader(set14_test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True,)

        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)
            
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing     TODO: 여기서 set 5, set 12, multi scale에 대해 결과를 뽑아야 함. 
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                # -------------------------------
                # set 5 measure
                # ------------------------------- 


                for test_set_name in ['set5', 'set14']:
                    if test_set_name == 'set5':
                        test_loader = set5_test_loader
                    elif test_set_name == 'set14':
                        test_loader = set14_test_loader

                    for scale in [2, 3, 4]:
                        avg_psnr = 0.0
                        avg_psnr_y = 0.0
                        idx = 0
                        
                        for test_data in test_loader:

                            idx += 1


                            ################################################
                            # TODO: 이거 나중에 제외 시키자. bicubic이 continuous하게 작동안하는 거 같아 임시적으로 넣음.
                            img_H = util.modcrop(test_data['H'].squeeze(), scale, channel_position=0)
                            img_H = torch.from_numpy(img_H).unsqueeze(0)
                            img_L = imresize.imresize(img_H, sizes=(img_H.shape[-2]//scale, img_H.shape[-1]//scale))

                            test_data['H'] = img_H
                            test_data['L'] = img_L
                            ################################################

                            model.feed_data(test_data)
                            model.test()

                            visuals = model.current_visuals()
                            E_img = util.tensor2uint(visuals['E'])
                            H_img = util.tensor2uint(visuals['H'])


                            # -----------------------
                            # calculate PSNR and PSNRY
                            # -----------------------

                            current_psnr = util.calculate_psnr(E_img, H_img, border=0)
                            current_psnr_y = util.calculate_psnr_y(E_img, H_img, border=0)
                            #logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                            avg_psnr += current_psnr
                            avg_psnr_y += current_psnr_y

                        avg_psnr = avg_psnr / idx
                        avg_psnr_y = avg_psnr_y / idx 

                        # testing log
                        logger.info('[{}, scale:{}] <epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average PSNR_Y : {:<.2f}db'.format(test_set_name, scale, epoch, current_step, avg_psnr, avg_psnr_y))


                    print("##########################################")
                print("##########################################" * 3)


            # -------------------------------
            # set 14 measure
            # ------------------------------- 



if __name__ == '__main__':
    main()
