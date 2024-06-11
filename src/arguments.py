import argparse


def parse():
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--output_path', default='./output', help='saved logging and dir path')
    parser.add_argument('--do_train', action='store_true', help='Do training')
    parser.add_argument('--do_eval', action='store_true', help='Do validation')
    parser.add_argument('--do_test', action='store_true', help='Do evaluate model')
    parser.add_argument('--device', default='auto', help='cuda device, i.e. 0 or 0,1,2,3 or auto')
    parser.add_argument('--accelerator', default='auto', help='cpu, gpu, tpu or auto')
    parser.add_argument('--strategy', default='auto', help='ddp, fsdp or auto')
    
    # data args
    parser.add_argument('--train_path', default=None, help='train folder path')
    parser.add_argument('--test_path', default=None, help='test folder path')
    parser.add_argument('--emotion_labels_train_path', default=None, help='emotion label train file')
    parser.add_argument('--emotion_labels_test_path', default=None, help='emotion label test file')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size training and  inference")
    
    # optimizer args
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--warmup', action='store_true', help='Do warmup steps')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs training')
    
    opt = parser.parse_args()
    
    return opt