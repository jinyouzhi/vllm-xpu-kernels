#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/util/Exception.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace {

void checkLevelZeroResult(ze_result_t result, const char* operation) {
  TORCH_CHECK(
      result == ZE_RESULT_SUCCESS,
      operation,
      " failed with Level Zero error code ",
      static_cast<uint32_t>(result));
}

size_t getTotalMemory(ze_device_handle_t device) {
  uint32_t memory_count = 0;
  checkLevelZeroResult(
      zeDeviceGetMemoryProperties(device, &memory_count, nullptr),
      "zeDeviceGetMemoryProperties");

  std::vector<ze_device_memory_properties_t> memory_properties(memory_count);
  for (auto& properties : memory_properties) {
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    properties.pNext = nullptr;
  }
  checkLevelZeroResult(
      zeDeviceGetMemoryProperties(
          device, &memory_count, memory_properties.data()),
      "zeDeviceGetMemoryProperties");

  size_t total_memory = 0;
  for (const auto& properties : memory_properties) {
    total_memory += properties.totalSize;
  }
  return total_memory;
}

std::optional<size_t> getUsableMemory(
    [[maybe_unused]] ze_driver_handle_t driver,
    [[maybe_unused]] ze_device_handle_t device) {
#ifdef ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME
  uint32_t extension_count = 0;
  checkLevelZeroResult(
      zeDriverGetExtensionProperties(driver, &extension_count, nullptr),
      "zeDriverGetExtensionProperties");

  std::vector<ze_driver_extension_properties_t> extensions(extension_count);
  checkLevelZeroResult(
      zeDriverGetExtensionProperties(
          driver, &extension_count, extensions.data()),
      "zeDriverGetExtensionProperties");
  const bool extension_available =
      std::any_of(extensions.begin(), extensions.end(), [](const auto& ext) {
        return std::strcmp(
                   ext.name, ZE_DEVICE_USABLEMEM_SIZE_PROPERTIES_EXT_NAME) == 0;
      });
  if (!extension_available) {
    return std::nullopt;
  }

  ze_device_properties_t device_properties{};
  ze_device_usablemem_size_ext_properties_t usable_memory_properties{};

  usable_memory_properties.stype =
      ZE_STRUCTURE_TYPE_DEVICE_USABLEMEM_SIZE_EXT_PROPERTIES;
  device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  device_properties.pNext = &usable_memory_properties;

  checkLevelZeroResult(
      zeDeviceGetProperties(device, &device_properties),
      "zeDeviceGetProperties");
  return usable_memory_properties.currUsableMemSize;
#else
  return std::nullopt;
#endif
}

size_t getFreeMemory(
    ze_driver_handle_t driver,
    ze_device_handle_t device,
    c10::DeviceIndex device_index) {
  if (const auto usable_memory = getUsableMemory(driver, device)) {
    return *usable_memory;
  }

  // A custom op can run before torch.xpu initializes its caching allocator.
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);
  return at::accelerator::getMemoryInfo(device_index).first;
}

}  // namespace

std::tuple<int64_t, int64_t> getMemoryInfo(int64_t device_index) {
  const auto index = static_cast<c10::DeviceIndex>(device_index);
  const auto& device = c10::xpu::get_raw_device(index);
  auto level_zero_device =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device);
  auto level_zero_driver =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
          device.get_platform());
  const size_t free =
      getFreeMemory(level_zero_driver, level_zero_device, index);
  const size_t total = getTotalMemory(level_zero_device);
  TORCH_CHECK(
      total <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
          free <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
      "XPU memory size exceeds int64_t");
  return {static_cast<int64_t>(free), static_cast<int64_t>(total)};
}
