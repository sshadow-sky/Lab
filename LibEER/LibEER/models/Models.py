# from models.CoralDgcnn import CoralDgcnn
# from models.DGCNN import DGCNN
# from models.DannDgcnn import DannDgcnn
# from models.RGNN import RGNN
from models.RGNN_official import SymSimGCNNet
from models.EEGNet import EEGNet
# from models.STRNN import STRNN
from models.GCBNet import GCBNet
from models.DBN import DBN
from models.TSception import TSception
# from models.SVM import SVM
from models.CDCN import CDCN
from models.HSLT import HSLT
from models.ACRNN import ACRNN
from models.GCBNet_BLS import GCBNet_BLS
from models.PCLTDGCN import PCLTDGCN
from models.FAT import FAT
from models.DSAGC import DSAGC
from models.FDGCL import FDGCL
from models.MAET import MAETForLibEER
from models.DMMR import DMMRPreTrainingModel, DMMRFineTuningModel, DMMRTestModel
from models.DMMR_GATTF import (
    DMMRGATTransformerFineTuningModel,
    DMMRGATTransformerPreTrainingModel,
    DMMRGATTransformerTestModel,
)
# from models.MsMda import MSMDA
from models.R2GSTNN import R2GSTNN
from models.BiDANN import BiDANN
# from models.FBSTCNet import PowerAndConneMixedNet
# from models.PRRL import PRRL
# from models.NSAL_DGAT import Domain_adaption_model
from models.PAA import PAA
from models.Sera import SERA

Model = {
    # 'DGCNN': DGCNN,
    # 'CoralDgcnn': CoralDgcnn,
    # 'DannDgcnn': DannDgcnn,
    'R2GSTNN': R2GSTNN,
    'BiDANN': BiDANN,
    'RGNN_official': SymSimGCNNet,
    'GCBNet': GCBNet,
    'GCBNet_BLS': GCBNet_BLS,
    'CDCN': CDCN,
    'DBN': DBN,
    # 'STRNN': STRNN,
    'EEGNet': EEGNet,
    'HSLT': HSLT,
    'ACRNN': ACRNN,
    'TSception': TSception,
    'PCLTDGCN': PCLTDGCN,
    'FAT': FAT,
    'DSAGC': DSAGC,
    'FDGCL': FDGCL,
    'MAET': MAETForLibEER,
    'DMMRPreTraining': DMMRPreTrainingModel,
    'DMMRFineTuning': DMMRFineTuningModel,
    'DMMRTest': DMMRTestModel,
    'DMMRGATTFPreTraining': DMMRGATTransformerPreTrainingModel,
    'DMMRGATTFFineTuning': DMMRGATTransformerFineTuningModel,
    'DMMRGATTFTest': DMMRGATTransformerTestModel,
    'PAA': PAA,
    'Sera': SERA,
    # 'MsMda': MSMDA,
    # "FBSTCNet": PowerAndConneMixedNet,
    # "NSAL_DGAT": Domain_adaption_model,
    # "PRRL" : PRRL,
    # 'svm' : SVM,

}
