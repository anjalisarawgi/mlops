import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

prof = profile(activities=[ProfilerActivity.CPU,  torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True)
# prof.start()
# model(inputs)
# prof.stop()   

with prof as prof:
    for i in range(10):
        model(inputs)
        prof.step()
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    # model(inputs)


# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     model(inputs)


# # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


prof.export_chrome_trace("trace.json")
