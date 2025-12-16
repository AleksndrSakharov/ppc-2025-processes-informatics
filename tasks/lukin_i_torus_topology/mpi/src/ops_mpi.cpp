#include "lukin_i_torus_topology/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lukin_i_torus_topology/common/include/common.hpp"

namespace lukin_i_torus_topology {

LukinIThorTopologyMPI::LukinIThorTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  OutType out = std::make_tuple(std::vector<int>{}, std::vector<int>{});
  GetOutput() = out;
}

bool LukinIThorTopologyMPI::ValidationImpl() {
  int start_v = std::get<0>(GetInput());
  int end_v = std::get<1>(GetInput());

  if (start_v < 0 || end_v < 0) {
    return false;
  }

  int proc_count = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  return start_v <= proc_count - 1 && end_v <= proc_count - 1;
}

bool LukinIThorTopologyMPI::PreProcessingImpl() {
  start_ = std::get<0>(GetInput());
  end_ = std::get<1>(GetInput());
  return true;
}

bool LukinIThorTopologyMPI::RunImpl() {
  int rank = -1;
  int proc_count = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  std::vector<int> message;
  int message_len = -1;

  if (rank == start_) {
    message = std::get<2>(GetInput());
    message_len = static_cast<int>(message.size());
  }

  if (HandleTrivial(message_len, message, proc_count)) {
    return true;
  }

  int cols = -1;
  int rows = -1;
  InitTopology(cols, rows, proc_count);

  int x = rank % cols;
  int y = rank / cols;

  int up = (((y - 1 + rows) % rows) * cols) + x;
  int down = (((y + 1) % rows) * cols) + x;
  int left = (y * cols) + ((x - 1 + cols) % cols);
  int right = (y * cols) + ((x + 1) % cols);
  std::unordered_map<Direction, int> dir_mapping = {
      {Direction::kUp, up}, {Direction::kDown, down}, {Direction::kLeft, left}, {Direction::kRight, right}};

  int source = start_;
  int dest = -1;
  std::vector<int> full_route;
  int route_size = -1;

  int end_x = end_ % cols;
  int end_y = end_ / cols;

  while (source != end_) {
    if (rank == source) {
      Direction direction = GetDir(x, y, end_x, end_y, cols, rows);
      dest = dir_mapping[direction];
    }
    MPI_Bcast(&dest, 1, MPI_INT, source, MPI_COMM_WORLD);
    if (rank == source) {
      Send(message_len, message, full_route, route_size, dest, rank);
    } else if (rank == dest) {
      Recieve(message_len, message, full_route, route_size, source);
    }
    source = dest;
  }

  if (rank == end_) {
    full_route.push_back(rank);
    route_size = static_cast<int>(full_route.size());
  }

  MPI_Bcast(&route_size, 1, MPI_INT, end_, MPI_COMM_WORLD);
  full_route.resize(route_size);
  MPI_Bcast(full_route.data(), route_size, MPI_INT, end_, MPI_COMM_WORLD);
  MPI_Bcast(&message_len, 1, MPI_INT, end_, MPI_COMM_WORLD);
  message.resize(message_len);
  MPI_Bcast(message.data(), message_len, MPI_INT, end_, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(full_route, message);
  return true;
}

LukinIThorTopologyMPI::Direction LukinIThorTopologyMPI::GetDir(int sx, int sy, int dx, int dy, int cols, int rows) {
  int mx = dx - sx;
  int my = dy - sy;

  if (abs(mx) > cols / 2) {
    mx = mx > 0 ? mx - cols : mx + cols;
  }
  if (abs(my) > rows / 2) {
    my = my > 0 ? my - rows : my + rows;
  }

  if (abs(mx) > abs(my)) {
    return mx < 0 ? Direction::kLeft : Direction::kRight;
  }

  return my < 0 ? Direction::kUp : Direction::kDown;
}

bool LukinIThorTopologyMPI::PostProcessingImpl() {
  return true;
}

void LukinIThorTopologyMPI::Send(int &message_len, std::vector<int> &message, std::vector<int> &full_route,
                                 int &route_size, int dest, int rank) {
  MPI_Send(&message_len, 1, MPI_INT, dest, static_cast<int>(Tags::kMlen), MPI_COMM_WORLD);
  MPI_Send(message.data(), message_len, MPI_INT, dest, static_cast<int>(Tags::kMessage), MPI_COMM_WORLD);
  full_route.push_back(rank);
  route_size = static_cast<int>(full_route.size());
  MPI_Send(&route_size, 1, MPI_INT, dest, static_cast<int>(Tags::kRoutesize), MPI_COMM_WORLD);
  MPI_Send(full_route.data(), route_size, MPI_INT, dest, static_cast<int>(Tags::kRoute), MPI_COMM_WORLD);
}

void LukinIThorTopologyMPI::Recieve(int &message_len, std::vector<int> &message, std::vector<int> &full_route,
                                    int &route_size, int source) {
  MPI_Recv(&message_len, 1, MPI_INT, source, static_cast<int>(Tags::kMlen), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  message.resize(message_len);
  MPI_Recv(message.data(), message_len, MPI_INT, source, static_cast<int>(Tags::kMessage), MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&route_size, 1, MPI_INT, source, static_cast<int>(Tags::kRoutesize), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  full_route.resize(route_size);
  MPI_Recv(full_route.data(), route_size, MPI_INT, source, static_cast<int>(Tags::kRoute), MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
}

bool LukinIThorTopologyMPI::HandleTrivial(int &message_len, std::vector<int> &message, int proc_count) {
  if (start_ == end_) {
    MPI_Bcast(&message_len, 1, MPI_INT, end_, MPI_COMM_WORLD);
    message.resize(message_len);
    MPI_Bcast(message.data(), message_len, MPI_INT, end_, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  if (proc_count == 1) {
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  return false;
}

void LukinIThorTopologyMPI::InitTopology(int &cols, int &rows, int proc_count) {
  for (rows = static_cast<int>(std::sqrt(static_cast<double>(proc_count))); rows > 0; rows--) {
    if (proc_count % rows == 0) {
      cols = proc_count / rows;
      break;
    }
  }
}

}  // namespace lukin_i_torus_topology
