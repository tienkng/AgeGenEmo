import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

import time
import onnx
import torch
import argparse
from modeling.emotic import EmoticNet
from utils import attem_load


DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='', help='metricnet weights path')
    parser.add_argument('--batch_size', type=int, default=1, help='saved logging and dir path')
    parser.add_argument("--simplify", action="store_true", help="simplify onnx model")
    parser.add_argument("--dynamic", action="store_true", help="dynamic ONNX axes")
    parser.add_argument("--im_size", type=int, default=224, help="Default ")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="True for using onnx_graphsurgeon to sort and remove unused",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="dynamic batch onnx for tensorrt and onnx-runtime",
    )
    
    opt = parser.parse_args()
    
    print("\nSummary: ", opt, "\n")

    t = time.time()
    img = torch.randn(opt.batch_size, 3, opt.im_size, opt.im_size)
    model = EmoticNet()
    model = attem_load(model, opt.checkpoint_path)
    
    with torch.no_grad():
        pred_age, pred_gender, pred_emotion = model(img)
    
    # output saved path
    saved_path = opt.checkpoint_path.replace(".ckpt", ".onnx")  # filename
    
    # Export model
    dynamic_axes = None
    if opt.dynamic:
        dynamic_axes = {"input": {0: "batch"}}
        output_axes = {
                "age" : {0: "batch", 1: "pred_age"},
                "gender" : {0: "batch", 2: "pred_gender"},
                "emotion" : {0: "batch", 3: "pred_emotion"}
            }
        dynamic_axes.update(output_axes)
        
    if opt.dynamic_batch:
        opt.batch_size = "batch"
        dynamic_axes = {
            "input": {0: "batch"},
        }
        output_axes = {
                "age" : {0: "batch"},
                "gender" : {0: "batch"},
                "emotion" : {0: "batch"}
            }
        dynamic_axes.update(output_axes)
        
    torch.onnx.export(
        model,  # model
        (img), # inputs
        saved_path,
        verbose=False,
        opset_version=12, 
        input_names = ['imgage'],
        output_names = ['age', 'gender', 'emotion'],
        dynamic_axes=dynamic_axes
    )
    
    # Checks
    onnx_model = onnx.load(saved_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx mode
    
    if opt.simplify:
        try:
            import onnxsim

            print("\nStarting to simplify ONNX...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
        except Exception as e:
            print(f"Simplifier failure: {e}")
            
    if opt.cleanup:
        try:
            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(onnx_model)
            graph = graph.cleanup().toposort()
            onnx_model = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")
            

    output =[node for node in onnx_model.graph.output]
    print('Outputs: ', output)
    
    onnx.save(onnx_model, saved_path)
    print("ONNX export success, saved as %s" % saved_path)

    # Finish
    print(
        "\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron."
        % (time.time() - t)
    )