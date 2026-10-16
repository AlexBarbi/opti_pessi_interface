#include "opti_pessi_interface/ObstacleSlots.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace opti_pessi {

std::vector<size_t> assignObstacleSlots(const std::vector<Eigen::Matrix<scalar_t, 2, 1>>& centres, std::vector<ObstacleSlot>& slots,
                                        scalar_t gate, int& nextId) {
  using Vec2 = Eigen::Matrix<scalar_t, 2, 1>;
  if (centres.size() > slots.size()) {
    throw std::invalid_argument("[assignObstacleSlots] more obstacles than slots");
  }

  std::vector<std::tuple<scalar_t, size_t, size_t>> pairs;  // distance, obstacle, slot
  for (size_t k = 0; k < centres.size(); ++k) {
    for (size_t j = 0; j < slots.size(); ++j) {
      const scalar_t distance = (centres[k] - Vec2(slots[j].x, slots[j].y)).norm();
      if (slots[j].taken && distance < gate) {
        pairs.emplace_back(distance, k, j);
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());

  constexpr size_t kNoSlot = std::numeric_limits<size_t>::max();
  std::vector<size_t> slotOf(centres.size(), kNoSlot);
  std::vector<int> idOf(centres.size(), -1);
  std::vector<bool> slotUsed(slots.size(), false);
  for (const auto& [distance, k, j] : pairs) {
    if (slotOf[k] == kNoSlot && !slotUsed[j]) {
      slotOf[k] = j;
      idOf[k] = slots[j].id;
      slotUsed[j] = true;
    }
  }
  for (const bool wasTaken : {false, true}) {
    for (size_t k = 0; k < centres.size(); ++k) {
      for (size_t j = 0; slotOf[k] == kNoSlot && j < slots.size(); ++j) {
        if (!slotUsed[j] && slots[j].taken == wasTaken) {
          slotOf[k] = j;
          idOf[k] = nextId++;
          slotUsed[j] = true;
        }
      }
    }
  }

  std::fill(slots.begin(), slots.end(), ObstacleSlot{});
  for (size_t k = 0; k < centres.size(); ++k) {
    slots[slotOf[k]] = ObstacleSlot{true, centres[k].x(), centres[k].y(), idOf[k]};
  }
  return slotOf;
}

}  // namespace opti_pessi
