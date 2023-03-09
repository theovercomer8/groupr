from pickle import UnpicklingError
import time
import logging
from typing import Tuple
import imagehash
import open_clip
import torch
from PIL import Image
import numpy as np
import cv2
import re
import os
import hashlib

class Worker:
    class Initter:
        def __call__(self, f_list, h_list):
            global features_list, hamm_list
            features_list = f_list
            hamm_list = h_list
        
        def close(self):
            global pool
            try:
                pool.close()
            except Exception as e:
                print(f'{e}')
            
            try:
                pool.terminate()
            except Exception as e:
                print(f'{e}')

            try:
                pool.join()
            except Exception as e:
                print(f'{e}')




    def __init__(self, f_list, h_list, event, debug, files, hashfunc, hashmethod, log_path, cache_path, clip_model_name, precision, device, clip_model_path) -> None:
        global features_list, hamm_list
        features_list = f_list
        hamm_list = h_list
        self.event = event
        self.e = time.time()
        self.clip_loaded = False
       
        self.debug = debug
        self.files = files
        self.hashmethod = hashmethod
        self.hashfunc = hashfunc
        self.log_path = log_path
        self.cache_path = cache_path
        self.clip_model_name = clip_model_name
        self.precision = precision
        self.device = device
        self.clip_model_path = clip_model_path

        if self.debug:
            pid = os.getpid()
            logging.basicConfig(filename=os.path.join(self.log_path,f'similarity_{self.e}_p{pid}.debug.txt'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    def hydrate(self):
        if len(self.files) > 1:
            imgs = [Image.open(file) for file in self.files]
            self.T = [self.image_to_features(img) for img in imgs]
            self.T = torch.mean(torch.stack(self.T,dim=0),dim=0,keepdim=True).squeeze(0)
        else:
            imgs = [Image.open(self.files[0])]
            self.T = self.image_to_features(imgs[0])

        self.hashes = [self.hashfunc(img) for img in imgs]

    def similarity(self, image1_features: torch.Tensor,image2_features: torch.Tensor) -> float:
            with torch.no_grad(), torch.cuda.amp.autocast():
                y = image1_features.T.view(image1_features.T.shape[1],image1_features.T.shape[0])
                similarity = torch.matmul(y,image2_features.T)
            return similarity[0][0].item()

    # def process(self, args):
    #     try:
    #         global shared_list
    #         i, data_entry = args
    #         if self.event.is_set() or data_entry[0] is None:
    #             shared_list[i] = 0
    #             return
            
    #         features = data_entry[0]
    #         t = self.similarity(self.config.T,features)
    #         logging.debug(f'Got similarity {t}')
    #         del features
    #         shared_list[i] = t
    #     except Exception as e:
    #         logging.error(f'Got exception processing {args}: {e}')
    
    def load_clip_model(self):
        start_time = time.time()
        logging.info(f'Config cache path: {self.cache_path}')
    
        clip_model_name, clip_model_pretrained_name = self.clip_model_name.split('/', 2)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=clip_model_pretrained_name, 
            precision=self.precision,
            device=self.device,
            jit=False,
            cache_dir=self.clip_model_path
        )
        self.clip_model.to(self.device).eval()
        
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        end_time = time.time()
        logging.info(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")
        self.clip_loaded = True
        self.hydrate()


    def get_cached_path(self,path,ext,prepend_cache_folder=True,hashed_folders=True):
        cp = self.cache_path 

        base = os.path.splitext(path)[0]
        base = re.sub(r'[:/ \\]','_',base)
        if prepend_cache_folder:
            if hashed_folders:
                h = hashlib.new('sha256')
                h.update(base.encode()) 
                h = h.hexdigest()[:3]
                cp = os.path.join(cp,h[0])
                os.makedirs(cp,exist_ok=True)
                cp = os.path.join(cp,h[1])
                os.makedirs(cp,exist_ok=True)
                cp = os.path.join(cp,h[2])
                os.makedirs(cp,exist_ok=True)
        return os.path.join(cp,base + ext) if prepend_cache_folder else base + ext

    def image_to_features(self, image: Image, idx = -1) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        if self.debug and idx == 0:
            logging.debug(f'PREPROCESSED\n{images}')
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            if self.debug and idx == 0:
                logging.debug(f'ENCODED\n{image_features}')
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if self.debug and idx == 0:
                logging.debug(f'NORMALIZED\n{image_features}')
        return image_features
    
    def __call__(self, args):
        if not self.clip_loaded:
            self.load_clip_model()
        global features_list, hamm_list
        idx,img_path = args
        if self.debug:
            pid = os.getpid()
            logging.basicConfig(filename=os.path.join(self.log_path,f'similarity_{self.e}_p{pid}.debug.txt'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        try:
            
            img = None
            name = os.path.splitext(img_path)[0] + '.pt'
            old_cached_name = self.get_cached_path(img_path,'.pt',hashed_folders=False)
            cached_name = self.get_cached_path(img_path,'.pt')
            hash_name = self.get_cached_path(img_path,f'_{self.hashmethod}.txt')
            features = None
            if os.path.exists(name):
                try:
                    features = torch.load(name,map_location=torch.device(self.device))
                    logging.debug(f'Loaded {name}')
                except UnpicklingError as e:
                    pass
            elif os.path.exists(old_cached_name):
                try:
                    features = torch.load(old_cached_name,map_location=torch.device(self.device))
                    logging.debug(f'Loaded {old_cached_name}')
                except UnpicklingError as e:
                    pass

            elif os.path.exists(cached_name):
                try:
                    features = torch.load(cached_name,map_location=torch.device(self.device))
                    logging.debug(f'Loaded {cached_name}')
                except UnpicklingError as e:
                    pass
            if features is None:
                img = Image.open(img_path).convert('RGB')
                features =  self.image_to_features(img,idx=idx)
                torch.save(features, cached_name)
                logging.debug(f'Saved {cached_name}')

            restored_hash = None
            if os.path.exists(hash_name):
                try:
                    with open(hash_name,'r') as f:
                        hash_as_str = f.read()
                    if self.hashmethod == 'crop-resistant':
                        restored_hash = imagehash.hex_to_multihash(hash_as_str)
                    elif self.hashmethod == 'colorhash':
                        restored_hash = imagehash.hex_to_flathash(hash_as_str, hashsize=3)
                    else:
                        restored_hash = imagehash.hex_to_hash(hash_as_str)
                except:
                    pass
            if restored_hash is None:
                if img is None:
                    img = Image.open(img_path).convert('RGB')

                restored_hash = self.hashfunc(img)
                with open(hash_name,'w') as f:
                    f.write(str(restored_hash))

            logging.debug(f'Image Hash: {restored_hash}')
            if idx == 0:
                logging.debug(f'IDX 0 TENSOR\n{features}')

        except Exception as e:
            logging.error(f'Could not load image path: {img_path}, error: {e}')


        try:
            if self.event.is_set() or features is None:
                features_list[idx] = 0
                hamm_list[idx] = 0
                return
            
            t = self.similarity(self.T,features)
            hamms = [h - restored_hash for h in self.hashes]
            hamm = np.average(hamms)
            del features, restored_hash
            features_list[idx] = t
            hamm_list[idx] = int(hamm)
        except Exception as e:
            logging.error(f'Got exception processing {args}: {e}')


# class TensorLoadingDataset(torch.utils.data.Dataset):
#     def __init__(self, config):
#         pass

#     def __len__(self):
#         return len(self.config.images)


#     def image_to_thumb(self, image: Image) -> Image:
#         img = image
#         img = np.array(image, np.uint8)

#         # Calculate max_pixels from max_resolution string
#         max_pixels = 256*256

#         # Calculate current number of pixels
#         current_pixels = img.shape[0] * img.shape[1]

#         # Check if the image needs resizing
#         if current_pixels > max_pixels:
#             smallest_side = img.shape[0] if img.shape[0] <= img.shape[1] else img.shape[1]
#             # Calculate scaling factor
#             scale_factor = 256 / smallest_side

#             # Calculate new dimensions
#             new_height = int(img.shape[0] * scale_factor)
#             new_width = int(img.shape[1] * scale_factor)

#             # Resize image
#             img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
#         else:
#             new_height, new_width = img.shape[0:2]

#         # Calculate the new height and width that are divisible by divisible_by (with/without resizing)
#         new_height = 256
#         new_width = 256

#         # Center crop the image to the calculated dimensions
#         y = int((img.shape[0] - new_height) / 2)
#         x = int((img.shape[1] - new_width) / 2)
#         img = img[y:y + new_height, x:x + new_width]

        
#         img = Image.fromarray(img)
#         return img
        
    

    




