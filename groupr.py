import argparse
from multiprocessing import set_start_method
import gradio as gr
import tqdm
import time
import sys
from helpers import groupr
from PIL import Image
Image.init()

config:groupr.Config = None

with gr.Blocks() as app:
    gr.HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/sortable/0.8.0/js/sortable.min.js"></script>')

    output = gr.Textbox(label="Status")
    folder = gr.Textbox(label="Dataset Path")
    uploaded = gr.Gallery(visible=False).style(grid=2)
    f = gr.Files(file_types=['image'])

    files_list = []
    def file_change(files):
        if files is None:
            return {uploaded: gr.update(visible=False)}
        global files_list
        files_list = files
        v = [file.name for file in files]
        return {uploaded: gr.update(visible=True, value = v)}
    f.change(file_change,inputs=f, outputs=uploaded)
    phash_type = gr.Dropdown(['ahash', 'phash','dhash','whash-haar','whash-db4','colorhash','crop-resistant'],value='crop-resistant',label="PHASH Method")
    clip_weight = gr.Slider(0,1.0,1.0,step=0.1,label="CLIP Weight")
    phash_weight = gr.Slider(0,1.0,1.0,step=0.1,label="PHASH Weight")
    go_btn = gr.Button("Goooooooo!")
    gal = gr.Gallery().style(grid=(2,3,4,5,6))

    with gr.Row() as row:
        html = gr.HTML()
    def track_tqdm(files_list, folder_path,hashmethod, clip_weight, phash_weight, progress=gr.Progress(track_tqdm=True)):
        global config
        config.files = files_list
        config.folder = folder_path
        config.hashmethod = hashmethod
        config.clip_weight = clip_weight
        config.phash_weight = phash_weight
        results = groupr.process(config)
        imgs = [Image.open(res[0]) for res in results]

        # h = f'''
        # <div id="thumbs">
        # '''
        # for 
        # H =+ f'''
        # </div>
        # '''
        return imgs, "Done", ""
    # with gr.Row(elem_id='gal_row')
    #     gal = gr.HTML()
    go_btn.click(track_tqdm, [f,folder,phash_type, clip_weight, phash_weight], outputs=[gal,output, html])


if __name__ == "__main__":
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers",type=int,default=4,help='Max workers to use for scanning. Lower to decerease VRAM usage. (default: 4)')
    parser.add_argument("--max_results",type=int,default=100,help='Max similar results to return. (default: 100)')
    parser.add_argument("--cache_path",type=str,default='./cache',help='Location to cache latents (default: ./cache)')
    parser.add_argument("--device",type=str,default='cuda',choices=['cuda','cpu'],help='Device (default: cuda)')
    parser.add_argument("--precision",type=str,default='fp16',choices=['fp16','bf16','fp32'],help="Floating point precision to use. Choose based on compatibility with your GPU. (default: fp16)")
    parser.add_argument("--debug",action="store_true",help='Location to cache latents (default: ./cache)')
    cfg = parser.parse_args()
    config = groupr.Config(max_workers=cfg.max_workers,
        max_results=cfg.max_results,
        cache_path=cfg.cache_path,
        debug=cfg.debug, 
        device=cfg.device, 
        precision=cfg.precision)
    app.queue() 
    app.launch()
