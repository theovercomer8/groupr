from ctypes import c_wchar
import json
import logging
from multiprocessing import Array, Pool, RawArray
import multiprocessing
import os
import time
import torch
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import tqdm
from .util import TensorLoadingDataset, Worker, load_clip_model
from concurrent import futures
@dataclass 
class Config:
    files = []
    folder:str = ''
    # models can optionally be passed in directly
    clip_model = None
    clip_preprocess = None

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: str = None
    base_dir: str = Path(os.path.dirname(__file__)).parent.absolute()
    # interrogator settings
    log_path: str = os.path.join(base_dir, 'log')
    cache_path: str = os.path.join(base_dir, 'cache')
    chunk_size: int = 2048
    data_path: str = os.path.join(base_dir, 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    quiet: bool = False # when quiet progress bars are not shown
    tokenize = None
    T:torch.Tensor = None
    max_workers:int = 8
    max_results:int = 100
    debug:bool = False
    precision:str = 'fp16'
    
def collate_fn_remove_corrupted(batch):
  """Collate function that allows to remove corrupted examples in the
  dataloader. It expects that the dataloader returns 'None' when that occurs.
  The 'None's in the batch are removed.
  """
  # Filter out all the Nones (corrupted examples)
  batch = list(filter(lambda x: x is not None, batch))
  return batch


# def similarity(image1_features: torch.Tensor,image2_features: torch.Tensor) -> float:
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             y = image1_features.T.view(image1_features.T.shape[1],image1_features.T.shape[0])
#             similarity = torch.matmul(y,image2_features.T)
#         return similarity[0][0].item()

def process(cfg:Config):
    global config
    e = time.time()
    config = cfg

    if config.debug:
        logging.basicConfig(filename=os.path.join(config.log_path,f'similarity_{e}.debug.txt'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    os.makedirs(config.cache_path,exist_ok=True)

    logging.debug(f'CONFIG:\n{config}')

    paths = []
    for root, dirs, files in os.walk(config.folder, topdown=False):
        for name in files:
            
            if os.path.splitext(os.path.split(name)[1])[1].upper() not in ['.JPEG','.JPG','.JPE', '.PNG', '.WEBP']:
                continue
            
            paths.append(os.path.join(root, name))

    if config.clip_model is None:
        load_clip_model(config)

    dataset = TensorLoadingDataset(paths, config.cache_path, config.clip_preprocess,config.device,config.clip_model)

    data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                       num_workers=config.max_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
    if len(config.files) > 1:
        imgs = [Image.open(file.name) for file in config.files]
        T = [dataset.image_to_features(img) for img in imgs]
        config.T = torch.mean(torch.stack(T,dim=0),dim=0,keepdim=True).squeeze(0)
    else:
        config.T = dataset.image_to_features(Image.open(config.files[0].name))

    features = Array('d', range(len(paths)), lock=False)
    m = multiprocessing.Manager()
    event = m.Event()
    w = Worker(features, config, event)
    initter=Worker.Initter()

    logging.debug(f'INPUT TENSOR\n{config.T}')

    # with futures.ThreadPoolExecutor(config.max_workers) as executor:
    with tqdm.tqdm(total=len(paths),desc="Processing") as pbar:
        for result in enumerate(data):
            if result[0] == 0:
                logging.debug(f'IDX 0 TENSOR\n{result[1][0]}')

            w.process(result)
            logging.debug(f'Processed {paths[result[0]]}')
        # for result in executor.map(w.process,enumerate(data)):
        # pool = Pool(config.max_workers, initializer=initter, initargs=(features,))
        # for result in pool.imap_unordered(w, enumerate(data), chunksize=64):
            pbar.update(1)
                
    
    
    d = {}
    for idx, p in enumerate(paths):
         d[p] = features[idx]

    a = sorted(d.items(), key=lambda x: x[1], reverse=True)[:config.max_results]
    json_object = json.dumps(a, indent=4)
    md = ''
    for item in a:
        md += f'### {item[1]}\n![{item[0]}]({item[0].replace(" ","%20")} "{item[1]}")\n\n\n'
    # Writing to sample.json

    os.makedirs(config.log_path,exist_ok=True)
    with open(os.path.join(config.log_path,f'similarity_{e}.json'), "w") as outfile:
        outfile.write(json_object)

    
    with open(os.path.join(config.log_path,f'similarity_{e}.md'), "w")  as outfile:
        outfile.write(md)

    del config.T
    return a