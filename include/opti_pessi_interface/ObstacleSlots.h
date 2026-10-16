#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "opti_pessi_interface/definitions.h"

namespace opti_pessi {

/** An obstacle as perceived now: centre (world/odom) and the keep-out model of its type. */
struct ObservedObstacle {
  scalar_t x = 0.0;
  scalar_t y = 0.0;
  scalar_t radius = 0.0;    // r_obs [m]
  scalar_t maxSpeed = 0.0;  // v_obs [m/s], the bound the pessimistic branch is robust against
};

/** The obstacle a slot held at the last assignment (centre and identity); a slot not taken was free. */
struct ObstacleSlot {
  bool taken = false;
  scalar_t x = 0.0;
  scalar_t y = 0.0;
  int id = -1;  // the same obstacle keeps its id from one assignment to the next; -1 while free
};

/**
 * Slot of each of `centres` (at most slots.size() of them), and `slots` updated to hold them.
 *
 * Each slot carries the obstacle's hyperplanes, which the solver warm-starts from the last solution, so an obstacle must
 * keep its slot, and its id, from one solve to the next. Pairs of an obstacle and a slot's previous centre closer than
 * `gate` are matched nearest first and keep that slot's id. The obstacles left over are new: they take a slot that was
 * free, then one whose obstacle is gone, lowest index first, each with the id `nextId` (then incremented). A slot no
 * obstacle takes becomes free.
 *
 * Pure Eigen: see test/testObstacleSlots.cpp.
 */
std::vector<size_t> assignObstacleSlots(const std::vector<Eigen::Matrix<scalar_t, 2, 1>>& centres, std::vector<ObstacleSlot>& slots,
                                        scalar_t gate, int& nextId);

}  // namespace opti_pessi
