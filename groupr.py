import argparse
from multiprocessing import set_start_method
import gradio as gr
import tqdm
import time
import sys
from helpers import groupr
from PIL import Image
import os
import shutil

Image.init()


config:groupr.Config = None
img_list = {}
sorted_list = None

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
    phash_type = gr.Dropdown(['ahash', 'phash','dhash','whash-haar','whash-db4','colorhash','crop-resistant'],value='phash',label="PHASH Method")
    clip_weight = gr.Slider(0,1.0,1.0,step=0.1,label="CLIP Weight")
    phash_weight = gr.Slider(0,1.0,1.0,step=0.1,label="PHASH Weight")
    max_results = gr.Slider(0,1000,100,step=10,label="Max Results")
    go_btn = gr.Button("Goooooooo!")
    gal = gr.Gallery().style(grid=(2,3,4,5,6))

    dest_path = gr.Textbox(label="Destination Path")
    mv_btn = gr.Button("Move Images & Associated Files")

    def move_imgs(dest_path, p_weight, c_weight, max_results):
        global img_list, config, sorted_list
        count = 0
        if sorted_list is None or len(sorted_list) == 0:
            return
        for f in sorted_list:
            path = f[0]
            path_no_ext = os.path.splitext(path)[0]

            filename = os.path.split(path)[1]
            shutil.move(path,os.path.join(dest_path,filename))
            count += 1
            if os.path.exists(path_no_ext + '.txt'):
                shutil.move(path_no_ext + '.txt',os.path.join(dest_path,os.path.splitext(filename)[0]) + '.txt')
                count += 1

            del img_list[path]

        return resort_list(p_weight, c_weight, max_results), f'{count} files moved'

    # with gr.Row() as row:
    #     html = gr.HTML()

    def resort_list(p_weight, c_weight, max_results):
        global img_list, config, sorted_list
        if img_list == {}:
            return []

        for p in img_list.keys():
            hamm = img_list[p][2]
            features = img_list[p][1]
            img_list[p] = ((features * c_weight) + (-0.1 * p_weight * hamm), features, hamm)
        
        sorted_list = sorted(img_list.items(), key=lambda x: x[1][0], reverse=True)[:max_results]
        imgs = []
        for img in sorted_list:
            try:

                img = Image.open(img[0])
                imgs.append(img)
            except:
                pass
        return imgs

    clip_weight.change(resort_list,inputs=[phash_weight,clip_weight,max_results],outputs=gal)
    phash_weight.change(resort_list,inputs=[phash_weight,clip_weight,max_results],outputs=gal)
    max_results.change(resort_list,inputs=[phash_weight,clip_weight,max_results],outputs=gal)

    mv_btn.click(move_imgs,inputs=[dest_path,phash_weight,clip_weight,max_results],outputs=[gal,output])

    def track_tqdm(files_list, folder_path,hashmethod, clip_weight, phash_weight, m_r, progress=gr.Progress(track_tqdm=True)):
        global config, img_list
        config.files = files_list
        config.folder = folder_path
        config.hashmethod = hashmethod
        config.clip_weight = clip_weight
        config.phash_weight = phash_weight
        config.max_results = m_r
        img_list = groupr.process(config)
        imgs = resort_list(phash_weight,clip_weight,config.max_results)

        # h = f'''
        # <div id="thumbs">
        # '''
        # for 
        # H =+ f'''
        # </div>
        # '''
        return imgs, "Done"
    # with gr.Row(elem_id='gal_row')
    #     gal = gr.HTML()
    go_btn.click(track_tqdm, [f,folder,phash_type, clip_weight, phash_weight, max_results], outputs=[gal,output])


if __name__ == "__main__":
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers",type=int,default=4,help='Max workers to use for scanning. Lower to decerease VRAM usage. (default: 4)')
    parser.add_argument("--cache_path",type=str,default='./cache',help='Location to cache latents (default: ./cache)')
    parser.add_argument("--device",type=str,default='cuda',choices=['cuda','cpu'],help='Device (default: cuda)')
    parser.add_argument("--precision",type=str,default='fp16',choices=['fp16','bf16','fp32'],help="Floating point precision to use. Choose based on compatibility with your GPU. (default: fp16)")
    parser.add_argument("--shared_clip",action="store_true", help="Use this to increase performance and allow less VRAM usage. Windows users may experience issues.")
    parser.add_argument("--debug",action="store_true",help='Location to cache latents (default: ./cache)')
    cfg = parser.parse_args()
    config = groupr.Config(max_workers=cfg.max_workers,
        cache_path=cfg.cache_path,
        debug=cfg.debug, 
        device=cfg.device, 
        precision=cfg.precision,
        shared_clip=cfg.shared_clip)
    app.queue() 
    app.launch()
