from multiprocessing import set_start_method
import gradio as gr
import tqdm
import time
import sys
from helpers import groupr
from PIL import Image
Image.init()


with gr.Blocks() as app:
    gr.HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/galleria/1.6.1/galleria.min.js"></script>')
    output = gr.Textbox(label="Status")
    folder = gr.Textbox(label="Dataset Path")
    
    f = gr.Files(file_types=['image'])

    files_list = []
    def file_change(files):
        global files_list
        files_list = files
    f.change(file_change,inputs=f)
    go_btn = gr.Button("Goooooooo!")
    gal = gr.Gallery().style(grid=(2,3,4,5,6))
    def track_tqdm(files_list, folder_path, progress=gr.Progress(track_tqdm=True)):
        results = groupr.process(files_list,folder_path)
        imgs = [Image.open(res[1][1]) for res in results]
        return imgs, "Done"
    # with gr.Row(elem_id='gal_row')
    #     gal = gr.HTML()
    go_btn.click(track_tqdm, [f,folder], outputs=[gal,output])

if __name__ == "__main__":
    set_start_method('spawn')
    app.queue() 
    app.launch()
