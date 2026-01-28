# https://nvidia.github.io/cuda-python/
import argparse
import os
import glob
def call_and_check_error(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert result[0] == 0, f"cuda-python error, {result[0]}"
        if len(result) == 2:
            return result[1]
        else:
            assert len(result) == 1, "Unsupported function call"
            return None
    return wrapper
def getCudaDeviceCount():
    from cuda import cudart
    return call_and_check_error(cudart.cudaGetDeviceCount)()
def getPCIBusIdByCudaDeviceOrdinal(cuda_device_id):
    '''
    cuda_device_id 在 0 ~ getCudaDeviceCount() - 1 之间取值，受到 CUDA_VISIBLE_DEVICES 影响
    '''
    from cuda import cuda
    device = call_and_check_error(cuda.cuDeviceGet)(cuda_device_id)
    result = call_and_check_error(cuda.cuDeviceGetPCIBusId)(100, device)
    return result.decode("ascii").split('\0')[0]
def get_dri_device_by_pci(pci_bus_id):
    '''
    通过 PCI 总线 ID 查找对应的 DRI 设备
    '''
    # 方法1: 遍历 /dev/dri/card* 设备，检查其 PCI 地址
    for card_device in glob.glob('/dev/dri/card*'):
        try:
            # 通过读取 sysfs 获取设备的 PCI 信息
            sysfs_path = f"/sys/class/drm/{os.path.basename(card_device)}/device"
            if os.path.exists(sysfs_path):
                real_path = os.path.realpath(sysfs_path)
                device_pci = os.path.basename(real_path)
                if device_pci.lower() == pci_bus_id.lower():
                    return card_device
        except:
            continue
    # 方法2: 如果找不到精确匹配，使用顺序映射
    # CUDA 设备顺序通常对应 /dev/dri/card 设备的顺序
    return f"/dev/dri/card{cuda_device_id}"
if __name__ == "__main__":
    # if os.environ.get("MINESTUDIO_GPU_RENDER", 0) != '1':
    #     print("cpu")
    #     exit(0)
    try:
        import cuda.cuda as cuda
        call_and_check_error(cuda.cuInit)(0)
    except:
        print("cpu")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=str)
    args = parser.parse_args()
    index = int(args.index)
    num_cuda_devices = getCudaDeviceCount()
    if num_cuda_devices == 0:
        device = "cpu"
    else:
        cuda_device_id = index % num_cuda_devices
        pci_bus_id = getPCIBusIdByCudaDeviceOrdinal(cuda_device_id)
        # 使用新的设备查找方法
        # device = get_dri_device_by_pci(pci_bus_id)
        device = "/dev/dri/card3"  # 修改为 card3
        # 验证设备是否存在
        if not os.path.exists(device):
            # 如果设备不存在，回退到顺序映射
            device = f"/dev/dri/card{cuda_device_id}"
            if not os.path.exists(device):
                # 如果还是不存在，使用第一个可用的设备
                available_cards = glob.glob('/dev/dri/card*')
                if available_cards:
                    device = available_cards[0]
                else:
                    device = "cpu"
    print(device)