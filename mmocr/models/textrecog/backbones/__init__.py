from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet31_ocr import ResNet31OCR
from .very_deep_vgg import VeryDeepVgg
from .resnet_extra import ResNetExtra
from .table_resnet_extra import TableResNetExtra
from .resnet50withGCB import ResNet50Extra
from .resnet50 import PretrainedResNet50
from .simple_projection import SimpleProjection

__all__ = ['ResNet31OCR', 'VeryDeepVgg', 'NRTRModalityTransform', 'ResNetExtra', 'TableResNetExtra','ResNet50Extra','PretrainedResNet50','SimpleProjection']