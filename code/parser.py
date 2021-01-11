import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Initial learning rate.')
    parser.add_argument( '-reg','--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--hidden', type=int, default=300,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping_rounds', type=int, default=30,    #rounds=10 also have value,more rounds maybe lead higher acc
                        help='early_stopping_rounds  ')

    args = parser.parse_args()
    return args