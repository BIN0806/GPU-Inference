#include "kvlab/rmsnorm_contract.hpp"
#include <gtest/gtest.h>
#include <array>
#include <limits>

namespace {
using kvlab::RmsNormValidation;
struct Fixture {
  alignas(8) std::array<unsigned char, 80> storage{};
  kvlab::RmsNormArgs args{1, 3, kvlab::RmsNormType::Float32,
                        {storage.data(), 12}, {storage.data() + 16, 12},
                        {storage.data() + 32, 12}, {storage.data() + 48, 12},
                        {storage.data() + 64, 12}};
};
TEST(RmsNormContract, ValidatesShapesTypesAndEmptyNoop) {
  Fixture f;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Valid);
  for (auto width : {0, -1, 8193}) {
    f.args.width = width;
    EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::InvalidShape);
  }
  f.args = {0, 8192, kvlab::RmsNormType::Float16, {}, {}, {}, {}, {}};
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Valid);
  f.args.rows = -1;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::InvalidShape);
  f.args.rows = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::InvalidShape);
  f.args.rows = 0;
  f.args.type = static_cast<kvlab::RmsNormType>(99);
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::UnsupportedType);
}
TEST(RmsNormContract, RejectsShortBuffersWithoutDereferencing) {
  Fixture f;
  f.args.y.bytes = 11;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::InsufficientCapacity);
  f.args.y.bytes = 12;
  f.args.weight.bytes = 8;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::InsufficientCapacity);
}
TEST(RmsNormContract, RejectsNullAndMisalignedBuffers) {
  Fixture f;
  f.args.x.data = nullptr;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::NullPointer);
  f.args.x.data = f.storage.data() + 1;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::MisalignedPointer);
  f.args.type = kvlab::RmsNormType::Float16;
  f.args.x.data = f.storage.data() + 2;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Valid);
}
TEST(RmsNormContract, RejectsAllKindsOfOverlap) {
  Fixture f;
  f.args.residual.data = f.args.x.data;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Overlap);
  f.args.residual.data = f.storage.data() + 8;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Overlap);
  f.args.residual.data = f.storage.data() + 16;
  f.args.y.data = f.args.z.data;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Overlap);
  f.args.y.data = f.storage.data() + 64;
  f.args.weight.data = f.args.z.data;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Overlap);
}
TEST(RmsNormContract, AcceptsAdjacentUsedRangesAndIgnoresUnusedCapacity) {
  Fixture f;
  f.args.x.bytes = 80;
  f.args.residual.data = f.storage.data() + 12;
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Valid);
}
TEST(RmsNormContract, RejectsWrappingAddressAndUsesWideElementCounts) {
  Fixture f;
  const auto max_address = std::numeric_limits<std::uintptr_t>::max();
  f.args.x.data = reinterpret_cast<const void*>(max_address - 3);
  EXPECT_EQ(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::AddressOverflow);
  f.args.x.data = f.storage.data();
  f.args.rows = 1LL << 30;
  f.args.width = 8192;
  // A wrapped 32-bit count could become zero and mistakenly accept tiny buffers.
  EXPECT_NE(kvlab::validate_rmsnorm_args(f.args), RmsNormValidation::Valid);
}
}  // namespace
