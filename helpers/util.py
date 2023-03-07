from pickle import UnpicklingError
import time
import logging
from typing import Tuple
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
        def __call__(self, the_list):
            global shared_list
            shared_list = the_list
        
        def close(args):
            global pool
            try:
                pool.close()
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f'{e}')



    def __init__(self, the_list, cfg, event) -> None:
        global shared_list
        shared_list = the_list
        self.config = cfg
        self.event = event

    def similarity(self, image1_features: torch.Tensor,image2_features: torch.Tensor) -> float:
            with torch.no_grad(), torch.cuda.amp.autocast():
                y = image1_features.T.view(image1_features.T.shape[1],image1_features.T.shape[0])
                similarity = torch.matmul(y,image2_features.T)
            return similarity[0][0].item()

    def process(self, args):
        try:
            global shared_list
            i, data_entry = args
            if self.event.is_set() or data_entry[0] is None:
                shared_list[i] = 0
                return
            
            features = data_entry[0]
            t = self.similarity(self.config.T,features)
            logging.debug(f'Got similarity {t}')
            del features
            shared_list[i] = t
        except Exception as e:
            logging.error(f'Got exception processing {args}: {e}')


class TensorLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, cache_path, clip_preprocess, device, clip_model, debug, log_path):
        self.images = image_paths
        self.cache_path = cache_path
        self.clip_preprocess = clip_preprocess
        self.device = device
        self.clip_model = clip_model
        self.log_path = log_path
        self.e = time.time()
        self.debug = debug
        self.clip_loaded = False
        if debug:
            pid = os.getpid()
            logging.basicConfig(filename=os.path.join(self.log_path,f'similarity_{self.e}_p{pid}.debug.txt'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

    def __len__(self):
        return len(self.images)

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

    def image_to_thumb(self, image: Image) -> Image:
        img = image
        img = np.array(image, np.uint8)

        # Calculate max_pixels from max_resolution string
        max_pixels = 256*256

        # Calculate current number of pixels
        current_pixels = img.shape[0] * img.shape[1]

        # Check if the image needs resizing
        if current_pixels > max_pixels:
            smallest_side = img.shape[0] if img.shape[0] <= img.shape[1] else img.shape[1]
            # Calculate scaling factor
            scale_factor = 256 / smallest_side

            # Calculate new dimensions
            new_height = int(img.shape[0] * scale_factor)
            new_width = int(img.shape[1] * scale_factor)

            # Resize image
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            new_height, new_width = img.shape[0:2]

        # Calculate the new height and width that are divisible by divisible_by (with/without resizing)
        new_height = 256
        new_width = 256

        # Center crop the image to the calculated dimensions
        y = int((img.shape[0] - new_height) / 2)
        x = int((img.shape[1] - new_width) / 2)
        img = img[y:y + new_height, x:x + new_width]

        
        img = Image.fromarray(img)
        return img
        
    def image_to_features(self, image: Image) -> torch.Tensor:
        if not self.clip_loaded:
            load_clip_model(self)
            self.clip_loaded = True
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def __getitem__(self, idx) -> torch.Tensor:
        if self.debug:
            pid = os.getpid()
            logging.basicConfig(filename=os.path.join(self.log_path,f'similarity_{self.e}_p{pid}.debug.txt'),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        try:
            
            img_path = self.images[idx]
            img = None
            name = os.path.splitext(img_path)[0] + '.pt'
            old_cached_name = self.get_cached_path(img_path,'.pt',hashed_folders=False)
            cached_name = self.get_cached_path(img_path,'.pt')
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
                features =  self.image_to_features(img)
                torch.save(features, cached_name)
                logging.debug(f'Saved {cached_name}')


            
            if idx == 0:
                logging.debug(f'IDX 0 TENSOR\n{features}')

        except Exception as e:
            logging.error(f'Could not load image path: {img_path}, error: {e}')
            return None

        return features


def load_clip_model(cfg):
        config = cfg
        start_time = time.time()
        logging.info(f'Config cache path: {config.cache_path}')
    
        clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
        config.clip_model, _, config.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, 
            pretrained=clip_model_pretrained_name, 
            precision=config.precision,
            device=config.device,
            jit=False,
            cache_dir=config.clip_model_path
        )
        config.clip_model.to(config.device).eval()
        
        config.tokenize = open_clip.get_tokenizer(clip_model_name)

        end_time = time.time()
        if not config.quiet:
            logging.info(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")

