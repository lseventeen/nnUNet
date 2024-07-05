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

def get_network_from_plans(plans_manager: PlansManager,
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

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
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
