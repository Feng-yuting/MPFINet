import argparse

parser=argparse.ArgumentParser(description='Pytorch Demo')
parser.add_argument('--dataset_train',type=str,default='../../dataset/Pansharpening_Dataset/QB/train')#train_dir
parser.add_argument('--dataset_val',type=str,default='../../dataset/Pansharpening_Dataset/QB/val')#val_dir
parser.add_argument('--batchSize',type=int,default=8,help='training batch size')
parser.add_argument('--net',type=str,default='MPFINet',help='network')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--log', default='log/', type=str)
parser.add_argument('--best_model', default='bestmodel_dir/', type=str)
parser.add_argument('--backup', default='backup_dir/', type=str)
parser.add_argument('--start_epoch', type=int, default=1,help='Manual epoch number (useful on restarts)')
parser.add_argument('--nEpochs', type=int, default=900, help='number of epochs to train for')
parser.add_argument("--resume", type=str,default="",  help="Path to checkpoint (default: none)")
parser.add_argument("--train_log", type=str,default="",  help="Path to save train log (default: none)")
parser.add_argument("--epoch_time_log", type=str,default="",  help="Path to save epoch time log (default: none)")
parser.add_argument("--val_log", type=str,default="",  help="Path to save validation log (default: none)")
parser.add_argument("--pretrained", type=str,default="",  help="path to pretrained model (default: none)")

parser.add_argument("--stem_16", type=int, default=16,  help="after stem the channel of PAN")
parser.add_argument("--stem_64", type=int, default=64,  help="after stem the channel of MS")
parser.add_argument("--stem_32", type=int, default=32,  help="after stem the channel of PAN")
parser.add_argument("--kernel_size", type=int, default=3,  help="the size of kernel_size")
parser.add_argument("--expansion", type=int, default=4,  help="the times of ConvBlock")
parser.add_argument("--image_size", type=int, default=256,  help="image_size")
parser.add_argument("--droprate", type=float, default=0.2,  help="droprate")
parser.add_argument("--channel_heads", type=tuple, default=[8,4,2],  help="MS Encoder Multi_heads")

opt = parser.parse_args()

