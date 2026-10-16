#include "opti_pessi_interface/OptiPessiReferenceManager.h"

#include <algorithm>
#include <numeric>

namespace opti_pessi {

size_t OptiPessiReferenceManager::setObservedObstacles(const std::vector<ObservedObstacle>& observed, const vector_t& robotCom) {
  using Vec2 = Eigen::Matrix<scalar_t, 2, 1>;
  const Vec2 com = robotCom.head<2>();
  const size_t numSlots = slots_.size();

  // With more obstacles than slots, the closest keep-out disks take them.
  std::vector<size_t> selected(observed.size());
  std::iota(selected.begin(), selected.end(), 0);
  if (selected.size() > numSlots) {
    const auto clearance = [&](size_t k) { return (Vec2(observed[k].x, observed[k].y) - com).norm() - observed[k].radius; };
    std::partial_sort(selected.begin(), selected.begin() + static_cast<std::ptrdiff_t>(numSlots), selected.end(),
                      [&](size_t a, size_t b) { return clearance(a) < clearance(b); });
    selected.resize(numSlots);
  }

  std::vector<Vec2> centres;
  centres.reserve(selected.size());
  for (const size_t k : selected) {
    centres.emplace_back(observed[k].x, observed[k].y);
  }
  const std::vector<size_t> slotOf = assignObstacleSlots(centres, slots_, kSlotGate, nextObstacleId_);

  for (size_t j = 0; j < numSlots; ++j) {
    obstacles_.row(static_cast<Eigen::Index>(j)) << com.x() + kParkingDistance, com.y();
    obstacleRadii_(static_cast<Eigen::Index>(j)) = 0.0;
    obstacleMaxSpeeds_(static_cast<Eigen::Index>(j)) = 0.0;
  }
  for (size_t k = 0; k < selected.size(); ++k) {
    const auto j = static_cast<Eigen::Index>(slotOf[k]);
    const ObservedObstacle& obstacle = observed[selected[k]];
    obstacles_.row(j) << obstacle.x, obstacle.y;
    obstacleRadii_(j) = obstacle.radius;
    obstacleMaxSpeeds_(j) = obstacle.maxSpeed;
  }
  return observed.size() - selected.size();
}

int OptiPessiReferenceManager::numActiveObstacles() const {
  return static_cast<int>(std::count_if(slots_.begin(), slots_.end(), [](const ObstacleSlot& slot) { return slot.taken; }));
}

namespace {

/** Tracking slots in the order an OCP's slots take them: occupied ones first, each group in slot order. */
std::vector<size_t> ocpSlotOrder(const std::vector<ObstacleSlot>& slots) {
  std::vector<size_t> order(slots.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_partition(order.begin(), order.end(), [&](size_t j) { return slots[j].taken; });
  return order;
}

}  // namespace

vector_t OptiPessiReferenceManager::getObstacleParameters(int numSlots) const {
  const std::vector<size_t> order = ocpSlotOrder(slots_);
  vector_t p(ObstacleP::DIM * numSlots);
  for (int i = 0; i < numSlots; ++i) {
    const int base = ObstacleP::DIM * i;
    if (static_cast<size_t>(i) < order.size()) {
      const auto j = static_cast<Eigen::Index>(order[static_cast<size_t>(i)]);
      p(base + ObstacleP::X) = obstacles_(j, 0);
      p(base + ObstacleP::Y) = obstacles_(j, 1);
      p(base + ObstacleP::RADIUS) = obstacleRadii_(j);
      p(base + ObstacleP::MAX_SPEED) = obstacleMaxSpeeds_(j);
      p(base + ObstacleP::ACTIVE) = slots_[order[static_cast<size_t>(i)]].taken ? 1.0 : 0.0;
    } else {
      // More OCP slots than tracking slots: free, parked where the free tracking slots are.
      p(base + ObstacleP::X) = obstacles_(0, 0);
      p(base + ObstacleP::Y) = obstacles_(0, 1);
      p(base + ObstacleP::RADIUS) = 0.0;
      p(base + ObstacleP::MAX_SPEED) = 0.0;
      p(base + ObstacleP::ACTIVE) = 0.0;
    }
  }
  return p;
}

std::vector<int> OptiPessiReferenceManager::getObstacleIds(int numSlots) const {
  const std::vector<size_t> order = ocpSlotOrder(slots_);
  std::vector<int> ids(static_cast<size_t>(numSlots), -1);
  for (size_t i = 0; i < ids.size() && i < order.size(); ++i) {
    ids[i] = slots_[order[i]].taken ? slots_[order[i]].id : -1;
  }
  return ids;
}

}  // namespace opti_pessi
