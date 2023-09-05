
from config.model_parameters import * 
from src.inference import load_model

def convert_to_onnx(onnx_file_name):

    model = load_model()

    example_input = torch.randn(1, 3, 50, 200).to(device)  # Adjust the shape according to your model's input requirements

     

    # Export the model to ONNX format
    torch.onnx.export(model,               # The PyTorch model
                    example_input,       # Example input data
                    onnx_file_name,      # Output ONNX file name
                    verbose=True,        # Print verbose information
                    input_names=['input'],   # List of input names
                    output_names=['output'], # List of output names
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # Dynamic axes (batch size)




if __name__ == '__main__':

    convert_to_onnx("model/model.onnx")