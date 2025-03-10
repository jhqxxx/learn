'''
Author: jhq
Date: 2025-03-04 18:32:08
LastEditTime: 2025-03-04 18:35:07
Description: 
'''
import torch
import torch.nn
import onnx
 
model = torch.load(r"D:\messy\ticket_detection_and_correction\pytorch_model.pt")
model.eval()
 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,3,768,768,requires_grad=True)
 
torch.onnx.export(model, x, r"D:\messy\ticket_detection_and_correction\pytorch_model.onnx", input_names=input_names, output_names=output_names, verbose='True')