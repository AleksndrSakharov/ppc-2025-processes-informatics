#include "telnov_transfer_one_all/seq/include/ops_seq.hpp"

namespace telnov_transfer_one_all {

template <typename T>
TelnovTransferOneAllSEQ<T>::TelnovTransferOneAllSEQ(const InType &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
  this->GetOutput().resize(0);
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::ValidationImpl() {
  return (!this->GetInput().empty()) && (this->GetOutput().empty());
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::PreProcessingImpl() {
  return true;
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::RunImpl() {
  this->GetOutput() = this->GetInput();
  return true;
}

template <typename T>
bool TelnovTransferOneAllSEQ<T>::PostProcessingImpl() {
  return !this->GetOutput().empty();
}

template class TelnovTransferOneAllSEQ<int>;
template class TelnovTransferOneAllSEQ<float>;
template class TelnovTransferOneAllSEQ<double>;

}  // namespace telnov_transfer_one_all
