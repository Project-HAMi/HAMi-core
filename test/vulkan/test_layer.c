#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

typedef VkResult (VKAPI_PTR *PFN_vkNegotiateLoaderLayerInterfaceVersion)(VkNegotiateLayerInterface*);

int main(void) {
    void *h = dlopen("./libvgpu.so", RTLD_NOW);
    assert(h != NULL);
    PFN_vkNegotiateLoaderLayerInterfaceVersion fn =
        (PFN_vkNegotiateLoaderLayerInterfaceVersion)
        dlsym(h, "vkNegotiateLoaderLayerInterfaceVersion");
    assert(fn != NULL);

    VkNegotiateLayerInterface iface = {0};
    iface.sType = LAYER_NEGOTIATE_INTERFACE_STRUCT;
    iface.loaderLayerInterfaceVersion = 2;
    VkResult r = fn(&iface);
    assert(r == VK_SUCCESS);
    assert(iface.pfnGetInstanceProcAddr != NULL);
    assert(iface.pfnGetDeviceProcAddr != NULL);
    printf("ok: layer entry point negotiates\n");
    return 0;
}
