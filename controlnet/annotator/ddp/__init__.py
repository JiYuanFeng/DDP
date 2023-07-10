import os
from mmseg.apis import inference_segmentor, init_segmentor,show_result_pyplot
from mmseg.core.evaluation import get_palette
from ..util import annotator_ckpts_path

class DDPDetector:
    def __ini__t (self):
        modelpath = os.path.join(annotator_ckpts_path, "ddp_swin_l_2x8_512x512_160k_ade20k.pth")
        assert os.path.exists(modelpath)
        config_file = os.path.join(os.path.dirname(annotator_ckpts_path), "ddp", "exp", "ddp_ade20k", "ddp_swin_l_2x8_512x512_160k_ade20k.py")
        self.model = init_segmentor(config_file, modelpath).cuda()
    def __call__ (self, img):
        result = inference_segmentor(self.model, img)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        colored_result = model.show_result(img, result, palette=get_palette('ade'), show=False, opacity=1)
        return colored_result

