import os
import cv2
import torch
import gradio as gr
import types
import numpy as np
import torchvision
from PIL import Image, ImageOps, ImageChops

from facexlib.utils.face_restoration_helper import FaceRestoreHelper

import modules.scripts as scripts

from modules import devices
import facexlib
from facexlib.detection import init_detection_model
import modules.sd_samplers as sd_samplers
import modules.sd_hijack_checkpoint as sd_hijack_checkpoint

from facexlib.detection.retinaface_utils import (decode, decode_landm, py_cpu_nms, PriorBox)

from modules.devices import NansException


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def init_img_with_mask(img):
    image, mask = img["image"], img["mask"]
    alpha = mask[:, :, 3]

    # Convert the alpha channel to a binary mask
    ret, binary_mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    # Convert the binary mask to a 3-channel mask
    mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, mask)
    return result


class Scripts(scripts.Script):
    def __init__(self) -> None:
        if hasattr(facexlib.detection.retinaface, 'device'):
            facexlib.detection.retinaface.device = devices.device
        model_rootpath = os.path.join(scripts.basedir(), 'models')
        if not os.path.exists(model_rootpath):
            os.makedirs(model_rootpath)
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png',
            device=devices.device, model_rootpath=model_rootpath)
        self.detector = init_detection_model('retinaface_resnet50', device=devices.device,
                                             model_rootpath=model_rootpath)
        self.segmentor = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(devices.device).eval()
        # Cache the original sampler creator.
        self.org_sampler = sd_samplers.create_sampler

    def title(self):
        return "Face Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # show in both modes
        ctrls = []
        # create a group for the UI elements
        with gr.Group():
            with gr.Accordion('Face Control', open=False):
                with gr.Row():
                    input_image = gr.Image(source='upload', mirror_webcam=False, type='numpy', tool='sketch')
                    extracted_face = gr.Image(label="Extracted Faces", visible=False)

                with gr.Row():
                    enabled = gr.Checkbox(label='Enable', value=False)
                    grad_weight = gr.Slider(label='Mix Weight', min=0, max=1, step=0.01, value=0.2, interactive=True)

                with gr.Row():
                    annotator_button = gr.Button(value="Preview extracted faces")
                    annotator_button_hide = gr.Button(value="Hide extracted faces")

                    def run_extractor(img):
                        if img is None:
                            return
                        img = init_img_with_mask(img)
                        result, landmarks , mask = self.extract_face(img)
                        if result is None:
                            return
                        mask = mask.astype(np.float)
                        mask = (mask - mask.min()) / (mask.max() - mask.min())
                        # Apply the mask to the original image
                        result = (mask * result) + ((1 - mask) * 255)
                        # Convert the result to uint8 format for display
                        result = np.clip(result, 0, 255).astype('uint8')
                        # Plot landmarks on the image
                        for landmark in landmarks:
                            cv2.circle(result, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)
                        return gr.update(value=result, visible=True, interactive=False)

                    annotator_button.click(fn=run_extractor, inputs=[input_image], outputs=[extracted_face])
                    annotator_button_hide.click(fn=lambda: gr.update(visible=False), inputs=None,
                                                outputs=[extracted_face])

                ctrls.append(enabled)
                ctrls.append(extracted_face)
                ctrls.append(input_image)
                ctrls.append(grad_weight)

        return ctrls

    def extract_face(self, img):
        with torch.no_grad():
            bboxes = self.detector.detect_faces(img, 0.97)
        if bboxes.shape[0] == 0:
            print("No faces detected")
            return None, None, None
        bboxes = bboxes[0]
        # expand the face region but don't go out of the image
        bboxes[0] = max(0, bboxes[0] - 0.1 * (bboxes[2] - bboxes[0]))
        bboxes[1] = max(0, bboxes[1] - 0.1 * (bboxes[3] - bboxes[1]))
        bboxes[2] = min(img.shape[1], bboxes[2] + 0.1 * (bboxes[2] - bboxes[0]))
        bboxes[3] = min(img.shape[0], bboxes[3] + 0.1 * (bboxes[3] - bboxes[1]))
        # crop the face region
        img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]
        landmarks = bboxes[5:]
        #  shift the landmarks
        landmarks = landmarks.reshape(-1, 2)
        landmarks[:, 0] -= bboxes[0]
        landmarks[:, 1] -= bboxes[1]
        # run the segmentation within the face region
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(devices.device).float() / 255.
        output = self.segmentor(img_tensor)['out']
        mask = output.argmax(1).squeeze().detach().cpu().numpy()
        mask = np.where(mask == 15, 255, 0).astype('uint8')
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        # Convert the mask to 3 channels
        mask = np.stack((mask,) * 3, axis=-1)
        return img, landmarks, mask

    def process(self, p, enabled, _, input_image_dict, mix_weight):
        if not enabled:
            # restore the sampling process
            sd_samplers.create_sampler = self.org_sampler
            sd_hijack_checkpoint.add()
            return
        sd_hijack_checkpoint.remove()

        if input_image_dict is None:
            return
        input_image = init_img_with_mask(input_image_dict)

        source_face, source_landmarks, source_mask = self.extract_face(input_image)
        if source_face is None:
            print("No source faces detected")
            return
        custom_sampler = self.org_sampler('DDIM', p.sd_model)
        device = devices.device

        mean_tensor = torch.tensor([[[[104.]], [[117.]], [[123.]]]]).to(device)
        face_detector = self.detector
        face_detector.resize = 1

        def detect_faces(inputs):
            # get scale
            height, width = inputs.shape[2:]
            face_detector.scale = torch.tensor([width, height, width, height], dtype=torch.float32).to(device)
            tmp = [width, height, width, height, width, height, width, height, width, height]
            face_detector.scale1 = torch.tensor(tmp, dtype=torch.float32).to(device)

            # forawrd
            inputs = inputs.to(device)
            if face_detector.half_inference:
                inputs = inputs.half()
            loc, conf, landmarks = face_detector(inputs)

            # get priorbox
            priorbox = PriorBox(face_detector.cfg, image_size=inputs.shape[2:])
            priors = priorbox.forward().to(device)

            return loc, conf, landmarks, priors

        @torch.no_grad()
        def ddim_custom(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                           temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                           unconditional_guidance_scale=1., unconditional_conditioning=None,
                           dynamic_threshold=None):

            b, *_, device = *x.shape, x.device
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                x_in = x
                model_output = self.model.apply_model(x_in, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
                elif isinstance(c, list):
                    c_in = list()
                    assert isinstance(unconditional_conditioning, list)
                    for i in range(len(c)):
                        c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            # pred_x0: (b, c, h, w)
            # decode the pred image
            img_tensor = 1. / self.model.scale_factor * pred_x0

            img_tensor = self.model.first_stage_model.decode(img_tensor)

            try:
                devices.test_for_nans(img_tensor, "decoded_image")
                x_sample_tensor = 255. * torch.clamp((img_tensor + 1.0) / 2.0, min=0.0, max=1.0)

                # debug: save the image to see if it is correct
                """
                x_sample = np.moveaxis(x_sample_tensor.squeeze(0).cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save("test_decoded_"+str(index)+".png")
                """

                # check if there is any faces in the tensor

                loc, conf, landmarks, priors = detect_faces(x_sample_tensor - mean_tensor)
                boxes = decode(loc.data.squeeze(0), priors.data, face_detector.cfg['variance'])
                boxes = boxes * face_detector.scale / face_detector.resize
                boxes = boxes.cpu().numpy()

                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landmarks = decode_landm(landmarks.squeeze(0), priors, face_detector.cfg['variance'])
                landmarks = landmarks * face_detector.scale1 / face_detector.resize
                landmarks = landmarks.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > 0.97)[0]
                boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

                # sort
                order = scores.argsort()[::-1]
                boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

                # do NMS
                bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(bounding_boxes, 0.4)
                bboxes, target_landmarks = bounding_boxes[keep, 0:4], landmarks[keep]

                if bboxes.shape[0] > 0 and index > 5:

                    target_landmarks = target_landmarks[0]
                    height, width = bboxes[0, 3] - bboxes[0, 1], bboxes[0, 2] - bboxes[0, 0]
                    # The height and width should be multiple of 8
                    height = int(height / 8) * 8
                    width = int(width / 8) * 8
                    # modify the bounding box to fit the height and width
                    bboxes[0, 0] = int(bboxes[0, 0])
                    bboxes[0, 1] = int(bboxes[0, 1])
                    bboxes[0, 2] = bboxes[0, 0] + width
                    bboxes[0, 3] = bboxes[0, 1] + height

                    # crop and transform the target landmarks
                    target_landmarks = target_landmarks.reshape(-1, 2)
                    target_landmarks[:, 0] = (target_landmarks[:, 0] - bboxes[0, 0])
                    target_landmarks[:, 1] = (target_landmarks[:, 1] - bboxes[0, 1])
                    # align the source face to the target face
                    transform = cv2.estimateAffinePartial2D(source_landmarks, target_landmarks)

                    # transform the source face
                    aligned = cv2.warpAffine(source_face, transform[0], (width, height))
                    mask_aligned = cv2.warpAffine(source_mask, transform[0], (width, height))
                    aligned_tensor = torch.from_numpy(aligned.astype(np.float32)).to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
                    aligned_tensor = 2 * aligned_tensor - 1

                    # encode the aligned face
                    aligned_latent = self.model.first_stage_model.encode(aligned_tensor).sample()
                    aligned_latent_ts = self.model.q_sample(aligned_latent, t)
                    latent_height, latent_width = aligned_latent_ts.shape[2], aligned_latent_ts.shape[3]
                    # resize the mask
                    mask_aligned = cv2.resize(mask_aligned, (latent_width, latent_height))
                    mask_aligned_tensor = torch.from_numpy(mask_aligned.astype(np.float32)).permute(2, 0, 1)[0].unsqueeze(0).unsqueeze(0).to(device) / 255.0 * mix_weight
                    # resize the bounding box to latent size (divided by 8)
                    bboxes /= 8
                    # fuse the aligned latent with the original latent x within the bounding box
                    # x: (b, c, h, w) , aligned_latent_ts: (b, c, height/8, width/8)
                    # mask_aligned_tensor: (b, 1, height/8, width/8)
                    x[:, :, int(bboxes[0, 1]):int(bboxes[0, 3]), int(bboxes[0, 0]):int(bboxes[0, 2])] = \
                        aligned_latent_ts * mask_aligned_tensor + x[:, :, int(bboxes[0, 1]):int(bboxes[0, 3]), int(bboxes[0, 0]):int(bboxes[0, 2])] * (1 - mask_aligned_tensor)

                    # recalculate e_t
                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                        x_in = x
                        model_output = self.model.apply_model(x_in, t, c)
                    else:
                        x_in = torch.cat([x] * 2)
                        t_in = torch.cat([t] * 2)
                        model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

                    if self.model.parameterization == "v":
                        e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
                    else:
                        e_t = model_output

                    if score_corrector is not None:
                        assert self.model.parameterization == "eps", 'not implemented'
                        e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

                    # recalculate pred_x0
                    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()


                    # debug code
                    # print(")
                    # cropped_face_tensor = 255. * torch.clamp(cropped_face.detach(), min=0.0, max=1.0)
                    # x_sample = np.moveaxis(cropped_face_tensor.squeeze(0).detach().cpu().numpy(), 0, 2).astype(np.uint8).copy()
                    # # draw the landmarks on the cropped face
                    # for i in range(5):
                    #     cv2.circle(x_sample, (int(target_landmarks[i, 0]), int(target_landmarks[i, 1])), 2, (0, 255, 0), -1)
                    # img = Image.fromarray(x_sample)
                    # img.save("test_cropped_"+str(index)+".png")
                    # img = Image.fromarray(aligned)
                    # img.save("test_aligned_"+str(index)+".png")

                    print("Faces detected! BBOX: ", bboxes)

            except Exception as e:
                print("Exception encountered. Skip:", e)
                pass

            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        custom_sampler.orig_p_sample_ddim = types.MethodType(ddim_custom, custom_sampler.sampler)

        def get_custom_sampler(name, model):
            print("Custom sampler created.")
            return custom_sampler

        sd_samplers.create_sampler = get_custom_sampler
        return p
