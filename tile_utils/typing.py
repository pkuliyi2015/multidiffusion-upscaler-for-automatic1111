import sys
from typing import *
NoType = Any

from torch import Tensor
from gradio.components import Component

from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from ldm.models.diffusion.ddpm import LatentDiffusion

from modules.processing import StableDiffusionProcessing as Processing, StableDiffusionProcessingImg2Img as ProcessingImg2Img, Processed
from modules.prompt_parser import MulticondLearnedConditioning, ScheduledPromptConditioning
from modules.extra_networks import ExtraNetworkParams
from modules.sd_samplers_kdiffusion import KDiffusionSampler, CFGDenoiser
# ↓↓↓ backward compatible for v1.5.2 ↓↓↓
try:
  from modules.shared_state import State
except ImportError:
  from modules.shared import State
try:
  from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion
except ImportError:
  CFGDenoiserKDiffusion = NoType
try:
  from modules.sd_samplers_timesteps import CompVisSampler, CFGDenoiserTimesteps, CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser
except ImportError:
  from modules.sd_samplers_compvis import VanillaStableDiffusionSampler
  CompVisSampler = VanillaStableDiffusionSampler
  CFGDenoiserTimesteps, CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser = NoType, NoType, NoType
# ↑↑↑ backward compatible for v1.5.2 ↑↑↑

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
