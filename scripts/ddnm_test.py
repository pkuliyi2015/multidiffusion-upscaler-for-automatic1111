import os
import cv2
import torch
import gradio as gr
import types
import numpy as np
from PIL import Image, ImageOps, ImageChops

import modules.scripts as scripts

from modules import devices
import modules.sd_samplers as sd_samplers
import modules.sd_hijack_checkpoint as sd_hijack_checkpoint

from modules.devices import NansException



class Scripts(scripts.Script):
    def __init__(self) -> None:
        model_rootpath = os.path.join(scripts.basedir(), 'models')
        if not os.path.exists(model_rootpath):
            os.makedirs(model_rootpath)
        # Cache the original sampler creator.

    def title(self):
        return "DDNM"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        if not is_img2img:
            return
        # show in both modes
        with gr.Group():
            with gr.Row():
                enabled = gr.Checkbox(label="Enable", value=False)
                scale = gr.Slider(label="Scaler", min=2, max=16, step=1, value=2)
        
        return[enabled, scale]

    def process(self, p, enabled, scale):
        if not enabled:
            # restore the sampling process
            sd_samplers.create_sampler = self.org_sampler
            return

        input_image = p.init_images[0]

        return p
