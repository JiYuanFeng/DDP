from functools import update_wrapper
from .densedepth_head import DenseDepthHead, DepthBaseDecodeHead
from .adabins_head import AdabinsHead
from .bts_head import BTSHead
from .dpt_head import DPTHead
from .binsformer_head import BinsFormerDecodeHead
from .newcrfs import NewCRFHead
from .deformable_head import DeformableHead
from .deformable_head_with_time import DeformableHeadWithTime
from .fcn_head import FCNHead