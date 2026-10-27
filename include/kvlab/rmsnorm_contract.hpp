#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kvlab {

enum class RmsNormType { Float16, Float32 };
struct ConstBufferView { const void* data = nullptr; std::size_t bytes = 0; };
struct BufferView { void* data = nullptr; std::size_t bytes = 0; };
struct RmsNormArgs {
  std::int64_t rows = 0;
  std::int64_t width = 0;
  RmsNormType type = RmsNormType::Float32;
  ConstBufferView x, residual, weight;
  BufferView z, y;
};

enum class RmsNormValidation {
  Valid, InvalidShape, UnsupportedType, SizeOverflow, InsufficientCapacity,
  NullPointer, MisalignedPointer, AddressOverflow, Overlap
};

// CPU-only descriptor checks. Never dereferences data or initializes CUDA.
// Buffer capacities and writable outputs are caller assertions, not allocation queries.
// An empty batch validates shape/type, then accepts empty buffers as a host no-op.
inline RmsNormValidation validate_rmsnorm_args(const RmsNormArgs& args) noexcept {
  if (args.rows < 0 || args.rows > std::numeric_limits<std::int32_t>::max() ||
      args.width < 1 || args.width > 8192) return RmsNormValidation::InvalidShape;
  std::size_t element_bytes = 0;
  switch (args.type) {
    case RmsNormType::Float16: element_bytes = 2; break;
    case RmsNormType::Float32: element_bytes = 4; break;
    default: return RmsNormValidation::UnsupportedType;
  }
  if (args.rows == 0) return RmsNormValidation::Valid;
  const auto width = static_cast<std::size_t>(args.width);
  const auto rows = static_cast<std::size_t>(args.rows);
  const auto max_size = std::numeric_limits<std::size_t>::max();
  if (rows > max_size / width / element_bytes) return RmsNormValidation::SizeOverflow;
  const std::size_t activation_bytes = rows * width * element_bytes;
  const std::array<ConstBufferView, 5> buffers = {
      args.x, args.residual, args.weight, {args.z.data, args.z.bytes}, {args.y.data, args.y.bytes}};
  std::array<std::uintptr_t, 5> starts{}, ends{};
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const auto required = i == 2 ? width * element_bytes : activation_bytes;
    if (buffers[i].bytes < required) return RmsNormValidation::InsufficientCapacity;
    if (buffers[i].data == nullptr) return RmsNormValidation::NullPointer;
    starts[i] = reinterpret_cast<std::uintptr_t>(buffers[i].data);
    if (starts[i] % element_bytes != 0) return RmsNormValidation::MisalignedPointer;
    if (required > std::numeric_limits<std::uintptr_t>::max() - starts[i])
      return RmsNormValidation::AddressOverflow;
    ends[i] = starts[i] + required;
    for (std::size_t j = 0; j < i; ++j) {
      if (starts[i] < ends[j] && starts[j] < ends[i]) return RmsNormValidation::Overlap;
    }
  }
  return RmsNormValidation::Valid;
}

}  // namespace kvlab
