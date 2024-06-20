import gradio as gr
from gradio_modal import Modal
import cv2
from PIL import Image, ImageDraw
import random
import numpy as np
from datetime import datetime
import os
import glob
import re
import scipy
from segment_anything import sam_model_registry, SamPredictor
import csv
from skimage.measure import label
from skimage.morphology import disk, binary_dilation


if not os.path.exists(f'annotation/gt'):
    os.makedirs(f'annotation/gt')

if not os.path.exists(f'annotation/images'):
    os.makedirs(f'annotation/images')

print('Load SAM Generator')
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def unshow_popup():
    return Modal(visible=False)

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def remove_small_noise(segmentation):
    labels = label(segmentation)
    labels_id = np.unique(labels)
    pixel_count = np.bincount(labels.flat)
    bin_count = {}
    for i in range(1, len(labels_id)):
        label_id = str(labels_id[i])
        pix_nb = pixel_count[i]
        bin_count[label_id]=pix_nb

    bin_count = {k: v for k, v in sorted(bin_count.items(), key=lambda item: item[1])}
    total = sum(bin_count.values())
    i = 1
    while np.sum([bin_count[x] for x in list(bin_count)[-i:]])/total<0.99:
        i = i +1

    largest_labels = [int(v) for v in list(bin_count)[-i:]]
    largestCC = labels == largest_labels[0]
    if len(largest_labels) >= 1:
        for i in range(1, len(largest_labels)):
            largestCC = largestCC + (labels == largest_labels[i])

    return largestCC

def postprocess_segm(mask):
    mask = mask.astype('uint8') * 255
    mask = binary_dilation(mask, disk(2))
    mask = scipy.ndimage.binary_fill_holes(mask).astype('uint8') * 255
    mask = remove_small_noise(mask).astype('uint8') * 255
    return mask

def open_csv():
    if os.path.isfile('annotation/masks_annotations.csv'):
        f_annot = open('annotation/masks_annotations.csv', 'a')
        writer_annot = csv.writer(f_annot)
    else:
        f_annot = open('annotation/masks_annotations.csv', 'w')
        # create the csv writer
        writer_annot = csv.writer(f_annot)
        header_annot = ['image_paths', 'gt_paths', 'label']
        writer_annot.writerow(header_annot)
    return writer_annot, f_annot

def add_to_list_input(coordinates_sam, evt):
    if coordinates_sam == '':
        list_input_points = [[evt[0], evt[1]]]
        coordinates_sam_text = f"{evt[0]},{evt[1]}"
    else:
        list_coord = coordinates_sam.split(";")
        if len(list_coord)==1:
            list_input_points = [list(np.asarray(list_coord[0].split(','), dtype=float)), [evt[0], evt[1]]]
        else:
            list_input_points = [list(np.asarray(v.split(','), dtype=float)) for v in list_coord] + [[evt[0], evt[1]]]
        coordinates_sam_text = coordinates_sam + f";{evt[0]},{evt[1]}"
    input_point = np.asarray(list_input_points)
    return input_point, coordinates_sam_text

def convert_text_input_coord(coordinates_sam):
    list_coord = coordinates_sam.split(";")
    if len(list_coord)==1:
        list_input_points = [list(np.asarray(list_coord[0].split(','), dtype=float))]
    else:
        list_input_points = [list(np.asarray(v.split(','), dtype=float)) for v in list_coord]
    input_point = np.asarray(list_input_points)
    return input_point

def get_select_coords(pic_img, back_img, coordinates_sam, name_file, mask_threshold, evt: gr.SelectData):
    if back_img is None:
        back_img = pic_img
        name_file = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        img_pic = Image.fromarray(back_img)
        img_pic.save(f'annotation/images/{name_file}.png')
    predictor.set_image(back_img)
    input_point, coordinates_sam = add_to_list_input(coordinates_sam, evt.index)
    input_label = np.ones(len(input_point))
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        return_logits = True,
    )
    mask = masks[0]
    np.save('mask_sam_segmentation_before_thresh.npy', mask)
    mask = mask > 0
    mask = postprocess_segm(mask)
    out = Image.fromarray(mask)
    pic_img = Image.fromarray(back_img)
    mask = Image.new("L", out.size, 100)
    out_im = Image.composite(out, pic_img, mask)


    for i in range(len(input_point)):
        n = input_point[i][0]
        x = input_point[i][1]

        draw = ImageDraw.Draw(out_im)
        size = 10
        draw.ellipse([n-size/2,x-size/2,n+size//2,x+size//2], fill="yellow")

    return out, out_im, back_img, coordinates_sam, name_file

def mask_changing_threshold(back_img, coordinates_sam, mask_threshold):
    input_point = convert_text_input_coord(coordinates_sam)
    mask = np.load('mask_sam_segmentation_before_thresh.npy')
    min_mask = np.min(mask)
    max_mask = np.max(mask)
    norm_mask = (mask-min_mask)/(max_mask - min_mask)
    mask = norm_mask > mask_threshold
    mask = postprocess_segm(mask)
    out = Image.fromarray(mask)
    pic_img = Image.fromarray(back_img)
    mask = Image.new("L", out.size, 100)
    out_im = Image.composite(out, pic_img, mask)


    for i in range(len(input_point)):
        n = input_point[i][0]
        x = input_point[i][1]

        draw = ImageDraw.Draw(out_im)
        size = 10
        draw.ellipse([n-size/2,x-size/2,n+size//2,x+size//2], fill="yellow")

    return out, out_im

def keep_mask_selected(saved_image, mask, name_file, trash_label):
    img_mask = Image.fromarray(mask)
    trash_label_filename = trash_label.replace(" ", "_")

    if not os.path.exists(f'annotation/gt/{trash_label_filename}'):
        os.makedirs(f'annotation/gt/{trash_label_filename}')
    
    fname = f'annotation/gt/{trash_label_filename}/{name_file}_{trash_label_filename}_1.png'
    if not os.path.isfile(fname):
        id_img = 1
        img_mask.save(fname)
    else:
        alist = glob.glob(f'annotation/gt/{trash_label_filename}/{name_file}_{trash_label_filename}_*.png')
        alist.sort(key=natural_keys)
        fname = alist[-1]
        id_img = str(int(fname.split('_')[-1][:-4]) + 1)
        fname = f'annotation/gt/{trash_label_filename}/{name_file}_{trash_label_filename}_{id_img}.png'
        img_mask.save(fname)


    writer_annot, f_annot = open_csv()
    annot_res = [f'{name_file}.png', f'{name_file}_{trash_label_filename}_{id_img}.png', trash_label_filename]
    writer_annot.writerow(annot_res)
    f_annot.close()

    image = gr.Image(label = 'Imagen.', value=saved_image)
    mask = gr.Image(label = 'Mask.', value=None)
    coordinates_sam = gr.Textbox(value=None)
    
    return image, mask, coordinates_sam, Modal(visible=False)

def setup_new_image():

    saved_image = gr.Image(label = 'Image.', value=None)
    image = gr.Image(label = 'Image.', value=None)
    mask = gr.Image(label = 'Image.', value=None)
    coordinates_sam = gr.Textbox(value=None)
    name_file = gr.Textbox(value=None)
   
    return saved_image, image, mask, coordinates_sam, name_file

def reset_mask(saved_image):

    image = gr.Image(label = 'Image.', value=saved_image)
    mask = gr.Image(label = 'Mask.', value=None)
    coordinates_sam = gr.Textbox(value=None)
    
    return image, mask, coordinates_sam


     
print('Running Gradio App...')
# Gradio UI

css = """
#keep {--button-secondary-background-fill: green}
#reset {--button-secondary-background-fill: red}
"""

with gr.Blocks(css=css) as demo:

    image = gr.Image(label = 'Image.', visible = True)
    mask = gr.Image(label = 'Mask.', visible = False)
    saved_image = gr.Image(label = 'Saved Image.', visible = False)
    coordinates_sam = gr.Textbox(value=None, visible=False)
    name_file = gr.Textbox(visible=False)

    trash_label = gr.Radio(["object_1", "object_2", "object_3", "object_4", "object_5"], label="Label", info="Choose the type of object.")

    with Modal(visible=False) as modal:
        gr.Markdown("""<center> <font size="+2"> <b> Are you sure? </b> </font> </center>""")
        with gr.Row():
            accept_bttn = gr.Button("Yes")
            cancel_bttn = gr.Button("No")

    with gr.Row():

        new_bttn = gr.Button('New Image', elem_id="new")
        new_bttn.click(setup_new_image, None, [saved_image, image, mask, coordinates_sam, name_file])

        keep_btn = gr.Button('Save Mask', elem_id="keep")

        reset_btn = gr.Button('Remove Mask', elem_id="reset")
        reset_btn.click(reset_mask, saved_image, [image, mask, coordinates_sam])

    keep_btn.click(lambda: Modal(visible=True), None, modal)
    accept_bttn.click(keep_mask_selected, [saved_image, mask, name_file, trash_label,], [image, mask, coordinates_sam, modal])
    cancel_bttn.click(unshow_popup, None, modal)

    with gr.Accordion("Advanced", open=False):
        
        mask_threshold = gr.Slider(
            label="Threshold segmentation score",
            interactive=True,
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.01,
        )
        mask_threshold.change(mask_changing_threshold, [saved_image, coordinates_sam, mask_threshold], [mask, image])

    image.select(get_select_coords, [image, saved_image, coordinates_sam, name_file, mask_threshold], [mask, image, saved_image, coordinates_sam, name_file])

demo.launch()