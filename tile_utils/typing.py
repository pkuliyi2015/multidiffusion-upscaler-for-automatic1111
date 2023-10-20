import sys
from typing import *
NoType = Any

from torch import Tensor
from gradio.components import Component

# NOTE: it is even ok(?) if the foundamentals are missing... :(
try:
  from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
  from ldm.models.diffusion.ddpm import LatentDiffusion
except ImportError:
  CompVisDenoiser, CompVisVDenoiser = NoType, NoType
  LatentDiffusion = NoType

# NOTE: it is ok if not the standard A1111/sd-webui repo :)
try:
  from modules.processing import StableDiffusionProcessing as Processing, StableDiffusionProcessingImg2Img as ProcessingImg2Img, Processed
  from modules.prompt_parser import MulticondLearnedConditioning, ScheduledPromptConditioning
  from modules.extra_networks import ExtraNetworkParams
  from modules.shared_state import State
  from modules.sd_samplers_kdiffusion import KDiffusionSampler, CFGDenoiserKDiffusion, CFGDenoiser
  from modules.sd_samplers_timesteps import CompVisSampler, CFGDenoiserTimesteps, CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser
except ImportError:
  Processing, ProcessingImg2Img, Processed = NoType, NoType, NoType
  MulticondLearnedConditioning, ScheduledPromptConditioning = NoType, NoType
  ExtraNetworkParams = NoType
  State = NoType
  KDiffusionSampler, CFGDenoiserKDiffusion, CFGDenoiser = NoType, NoType, NoType
  CompVisSampler, CFGDenoiserTimesteps, CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser = NoType, NoType, NoType, NoType

ModuleType = type(sys)

Sampler = Union[KDiffusionSampler, CompVisSampler]
Cond = MulticondLearnedConditioning
Uncond = List[List[ScheduledPromptConditioning]]
ExtraNetworkData = DefaultDict[str, List[ExtraNetworkParams]]

# 'c_crossattn'     List[Tensor[B, L=77, D=768]]    prompt cond (tcond)
# 'c_concat'        List[Tensor[B, C=5, H, W]]      latent mask (icond)
# 'c_adm'           Tensor[?]                       unclip (icond)
# 'crossattn'       Tensor[B, L=77, D=2048]         sdxl (tcond)
# 'vector'          Tensor[B, D]                    sdxl (tcond)
CondDict = Dict[str, Tensor]
