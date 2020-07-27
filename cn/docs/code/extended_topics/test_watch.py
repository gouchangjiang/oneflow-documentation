import numpy as np
import oneflow as flow

def cb(y):
    print("out", y.numpy())

@flow.global_function()
def ReluJob(x=flow.FixedTensorDef((10,))) -> None:
    y = flow.nn.relu(x)
    flow.watch(y, cb)

flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 10).astype(np.float32)
print("in: ", data)
ReluJob(data)
