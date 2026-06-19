from gnns import *
from pgformer import pgformer


def parse_method(args, d, c, d_para, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
        suffix = ''
    elif args.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'bn' + str(args.use_bn)
    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
        suffix = 'hops' + str(args.hops) + 'cached' + str(args.cached)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + '100alpha' + str(int(100*args.alpha))
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + '100alpha' + str(int(100*args.alpha))
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'bn' + str(args.use_bn) +\
            'heads' + str(args.gat_heads) + 'out_heads' + str(args.out_heads)
    elif args.method == 'egnn':
        model = EGNN(d, args.hidden_channels, c, num_layers=args.num_layers,
                     dropout=args.dropout, coord_channels=args.egnn_coord_channels,
                     use_bn=args.use_bn).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'bn' + str(args.use_bn) +\
            'coord' + str(args.egnn_coord_channels)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'hops' + str(args.hops)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'jk_type' + str(args.jk_type)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + 'heads' + str(args.gat_heads) + 'jk_type' + str(args.jk_type)
        
        
        
    elif args.method == 'pgformer':
        model=pgformer(d,args.hidden_channels, c, d_para, num_layers=args.num_layers, alpha=args.alpha, dropout=args.dropout, num_heads=args.num_heads, kernel=args.kernel,
                       use_bn=args.use_bn, use_residual=args.use_residual, use_graph=args.use_graph, use_weight=args.use_weight).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'layers' + str(args.num_layers) + '100alpha' + str(int(100*args.alpha)) + 'kernel' + args.kernel
        suffix += 'heads' + str(args.num_heads) + 'bn' + str(args.use_bn) + 'residual' + str(args.use_residual) + 'graph' + str(args.use_graph) + 'weight' + str(args.use_weight)

    
    
    elif args.method == 'mpnnlstm':
        model = MPNN_LSTM(d, args.hidden_channels, c, n, 1, args.dropout).to(device)
        suffix = 'hidden' + str(args.hidden_channels)
    elif args.method == 'dcrnn':
        model = DC_RNN(d, args.hidden_channels, c, args.dcrnn_filters).to(device)
        suffix = 'hidden' + str(args.hidden_channels) + 'filter' + str(args.dcrnn_filters)
    else:
        raise ValueError('Invalid method')
    return model, suffix

"""
taskset -c 14-20 python /mnt/HDD1/lc23/tyk/pgformer/src_pgformer/test2.py --id 56 --device 0 --epochs 1000 --batch_size 720     \
    --method pgformer --lr 0.001 --num_layers 2 --num_heads 4      \
    --hidden_channels 128 --dropout 0.6 --weight_decay 1e-5 --use_bn --use_residual --use_graph --save_model
"""

def parser_add_main_args(parser):
    parser.add_argument('--path', type=str, default='/root/autodl-tmp/tyk/pgformer')
    parser.add_argument('--id', type=int, default=0)
    
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/tyk/pgformer/data/mb.csv')
    parser.add_argument('--test_dir', type=str, default='/root/autodl-tmp/tyk/pgformer/data/test.csv')
    # parser.add_argument('--target', type=str, default='recall')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=720)
    
    # utilities
    parser.add_argument('--input', type=int, default=128, help='input channel')
    parser.add_argument('--output', type=int, default=3, help='output channel')
    parser.add_argument('--knn_num', type=int, default=2, help='number of k for KNN graph')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='/root/autodl-tmp/tyk/pgformer/model/')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    
    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # pgformer model
    parser.add_argument('--method', type=str, default='pgformer')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers for deep methods')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads for attention')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_graph', action='store_true', help='use pos emb')
    parser.add_argument('--use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--kernel', type=str, default='simple', choices=['simple', 'sigmoid'])

    # baseline model
    parser.add_argument('--egnn_coord_channels', type=int, default=3,
                        help='number of input feature channels used as EGNN coordinates')
    # parser.add_argument('--gat_heads', type=int, default=2,
    #                     help='attention heads for gat')
    # parser.add_argument('--out_heads', type=int, default=1,
    #                     help='out heads for gat')
    # parser.add_argument('--dcrnn_filters', type=int, default=1,
    #                     help='filter K for DCRNN')
    # parser.add_argument('--hops', type=int, default=1,
    #                     help='power of adjacency matrix for certain methods')
    # parser.add_argument('--lp_alpha', type=float, default=.1,
    #                     help='alpha for label prop')
    # parser.add_argument('--gpr_alpha', type=float, default=.1,
    #                     help='alpha for gprgnn')
    # parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
    #                     help='jumping knowledge type')

    

    


