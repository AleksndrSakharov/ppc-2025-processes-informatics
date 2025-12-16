#include "telnov_counting_the_frequency/seq/include/ops_seq.hpp"

#include <chrono>
#include <cstdint>
#include <string>

#include "telnov_counting_the_frequency/common/include/common.hpp"

namespace telnov_counting_the_frequency {

TelnovCountingTheFrequencySEQ::TelnovCountingTheFrequencySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TelnovCountingTheFrequencySEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool TelnovCountingTheFrequencySEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool TelnovCountingTheFrequencySEQ::RunImpl() {
  const std::string &s = GlobalData::g_data_string;
  int64_t cnt = 0;
  for (char c : s) {
    if (c == 'X') {
      cnt++;
    }
  }

  GetOutput() = static_cast<int>(cnt);

  using Clock = std::chrono::high_resolution_clock;
  auto delay_start = Clock::now();
  while (std::chrono::duration<double>(Clock::now() - delay_start).count() < 0.001) {
  }

  return true;
}

bool TelnovCountingTheFrequencySEQ::PostProcessingImpl() {
  return GetOutput() == GetInput();
}

}  // namespace telnov_counting_the_frequency
