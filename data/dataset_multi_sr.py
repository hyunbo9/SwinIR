import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetMultiSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetMultiSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):
        if self.opt['phase'] == 'train':
            return self.get_item_at_train(index)
        
        elif self.opt['phase'] == 'test':
            return self.get_item_at_test(index)

        else:
            raise Error("phase Error")

    def get_item_at_train(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        L_path = H_path
        img_L = None

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}
    
    def get_item_at_test(self, index):
        if not self.paths_L:
            raise Error("have to exist L image path, because this setting is multi scale")

        sf = 4  # TODO: sf를 2, 3, 4 전체 다에 대해서 구현. 
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, sf)   
        # ------------------------------------
        # get L image
        # ------------------------------------

        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)
        img_L = util.uint2single(img_L)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


class RandomResizeCollater(object):
    def __init__(self, patch_size, sf_list=[2, 3, 4]):
        self.patch_size = patch_size 
        self.sf_list = sf_list 

    def __call__(self, samples):
        


        sf = random.choice(self.sf_list)
        L_size = self.patch_size // sf

        img_Hs = []
        img_Ls = [] 
        L_pathes = []
        H_pathes = []
        for sample in samples:
            img_H = sample['H']

            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / sf, True)

            # ------------------------------------
            # if train, get L/H patch pair
            # ------------------------------------

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - L_size))
            rnd_w = random.randint(0, max(0, W - L_size))
            img_L = img_L[rnd_h:rnd_h + L_size, rnd_w:rnd_w + L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

            # ------------------------------------
            # L/H pairs, HWC to CHW, numpy to tensor
            # ------------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

            img_Ls.append(img_L)
            img_Hs.append(img_H)
            L_pathes.append(sample['L_path'])
            H_pathes.append(sample['H_path'])


        return {'L': torch.stack(img_Ls, axis=0), 
                'H': torch.stack(img_Hs, axis=0), 
                'L_path': L_pathes, 
                'H_path': H_pathes}

        