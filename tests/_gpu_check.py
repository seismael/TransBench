import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("Mem:", round(props.total_memory / 1e9, 1), "GB")
else:
    print("GPU: N/A")
