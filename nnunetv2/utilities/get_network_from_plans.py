<<<<<<< HEAD
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.network_architecture.phtransV2 import PHTrans
from nnunetv2.network_architecture.cotr.ResTranUnet import ResTranUnet
from nnunetv2.network_architecture.swin_unetr import SwinUNETR
from nnunetv2.network_architecture.nnformer import nnFormer
from nnunetv2.network_architecture.unetr import UNETR
from nnunetv2.network_architecture.UXNet_3D.network_backbone import UXNET 
from torch import nn
import torch
=======
import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join
>>>>>>> fee8c2db4a52405389eb5d3c4512bd2f654ab999


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )
<<<<<<< HEAD
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model



def get_custom_network_from_plans(custom_network: str,
                                  plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

  

    # network class name!!
    if custom_network == "PHTrans":
        model = PHTrans(img_size=configuration_manager.patch_size,  
                                base_num_features=32,
                                num_classes=label_manager.num_segmentation_heads, 
                                image_channels=num_input_channels,
                                only_conv=False,
                                num_conv_per_stage=1,
                                num_transformer_per_stage = 2,
                                pool_op_kernel_sizes=configuration_manager.pool_op_kernel_sizes[1:],
                                conv_kernel_sizes=configuration_manager.conv_kernel_sizes, 
                                deep_supervision=deep_supervision,
                                drop_path_rate=0.2,
                                dropout_p=0.1)
    elif custom_network == "UNETR":
        model = UNETR(
        in_channels=num_input_channels,
        out_channels=label_manager.num_segmentation_heads, 
        img_size=configuration_manager.patch_size,  
        feature_size=32,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        )
    elif custom_network == "Swin_UNETR":
        model = SwinUNETR(
        img_size=configuration_manager.patch_size,  
        in_channels=num_input_channels,
        out_channels=label_manager.num_segmentation_heads, 
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0,
        use_checkpoint=True,
        )
        weight = torch.load("/ai/code/nnUNet/nnunetv2/network_architecture/pretrain_model/model_swinvit.pt")
        model.load_from(weights=weight)
        print("Using pretrained self-supervied Swin UNETR backbone weights !")

    elif custom_network == "cotr": 
        model = ResTranUnet(
            norm_cfg="IN", 
            activation_cfg="LeakyReLU", 
            img_size=configuration_manager.patch_size,  
            num_classes=label_manager.num_segmentation_heads, 
            weight_std=False, 
            deep_supervision=deep_supervision)

    elif custom_network == "nnFormer": 
        model=nnFormer(crop_size=configuration_manager.patch_size,  
                            embedding_dim=192,
                            input_channels=num_input_channels,
                            num_classes=label_manager.num_segmentation_heads, 
                            conv_op=conv_op,
                            depths=[2, 2, 2, 2],
                            num_heads=[6, 12, 24, 48],
                            patch_size=[2,4,4],
                            window_size=[4,4,8,4],
                            deep_supervision=True)





    elif custom_network == '3DUXNET':
        model = UXNET(
                    in_chans=num_input_channels,
                    out_chans=label_manager.num_segmentation_heads, 
                    depths=[2, 2, 2, 2],
                    feat_size=[48, 96, 192, 384],
                    drop_path_rate=0,
                    layer_scale_init_value=1e-6,
                    spatial_dims=3,
                    )
    model.apply(InitWeights_He(1e-2))
    # if network_class == ResidualEncoderUNet:
    #     model.apply(init_last_bn_before_add_to_0)
    return model
=======

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network
>>>>>>> fee8c2db4a52405389eb5d3c4512bd2f654ab999
