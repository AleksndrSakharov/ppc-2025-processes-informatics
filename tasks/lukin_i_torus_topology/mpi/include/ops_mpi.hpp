#pragma once

#include <cstdint>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lukin_i_torus_topology {

class LukinIThorTopologyMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LukinIThorTopologyMPI(const InType &in);

 private:
  enum class Direction : std::uint8_t { kUp, kDown, kLeft, kRight, kNone };

  enum class Tags : std::uint8_t { kRoutesize, kRoute, kMessage, kMlen };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static Direction GetDir(int sx, int sy, int dx, int dy, int cols, int rows);

  static void Send(int &message_len, std::vector<int> &message, std::vector<int> &full_route, int &route_size, int dest,
                   int rank);
  static void Recieve(int &message_len, std::vector<int> &message, std::vector<int> &full_route, int &route_size,
                      int source);
  bool HandleTrivial(int &message_len, std::vector<int> &message, int proc_count);
  static void InitTopology(int &cols, int &rows, int proc_count);

  int start_ = -1;
  int end_ = -1;
};

}  // namespace lukin_i_torus_topology
