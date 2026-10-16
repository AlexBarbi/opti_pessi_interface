#pragma once

#include <vector>

#include <ocs2_oc/synchronized_module/ReferenceManager.h>

#include "opti_pessi_interface/ObstacleSlots.h"
#include "opti_pessi_interface/OptiPessiModelParameters.h"
#include "opti_pessi_interface/definitions.h"

namespace opti_pessi {

/**
 * Carries the per-iteration, non-differentiated data of the Opti-Pessi OCP: the goal, the measured
 * obstacles (centre, radius, speed bound and whether a slot holds one at all), the trot parity at
 * the start of the horizon, and the keep-out homotopy knob used by the closed loop when a solve fails.
 *
 * Cost and constraint terms hold a const reference to this object and read it inside their
 * getParameters(...) hooks, so a change here is picked up by the next solve without rebuilding any
 * CppAD library. This mirrors how legged_interface's constraints consume
 * SwitchedModelReferenceManager.
 */
class OptiPessiReferenceManager : public ocs2::ReferenceManager {
 public:
  explicit OptiPessiReferenceManager(const OptiPessiModelParameters& params)
      : ocs2::ReferenceManager(ocs2::TargetTrajectories(), ocs2::ModeSchedule()),
        goal_(params.goal),
        obstacles_(matrix_t::Zero(params.numObstacles(), 2)),
        obstacleRadii_(vector_t::Zero(params.numObstacles())),
        obstacleMaxSpeeds_(vector_t::Zero(params.numObstacles())),
        slots_(static_cast<size_t>(params.numObstacles())) {
    // Every slot free until the first setObservedObstacles().
    obstacles_.col(0).setConstant(kParkingDistance);
  }

  ~OptiPessiReferenceManager() override = default;

  /** Goal CoM position in the world frame. */
  const vector_t& getGoal() const { return goal_; }
  void setGoal(const vector_t& goal) { goal_ = goal; }

  /** Obstacle centres of the tracking slots (scenario obstacles.numSlots of them) x 2, free ones parked. */
  const matrix_t& getObstacles() const { return obstacles_; }

  /**
   * Fills the tracking slots (scenario obstacles.numSlots) from the obstacles perceived now, around a robot whose CoM is
   * at `robotCom`. Call it between solves, never during one.
   *
   * With more obstacles than slots, the ones whose keep-out disks come closest to `robotCom` take them. An obstacle
   * keeps its slot and its id from the previous call (see assignObstacleSlots(), gate kSlotGate), so the hyperplanes the
   * solver warm-started for it stay its own. A free slot's centre is parked kParkingDistance from the robot, which only
   * the hyperplane seed reads. Returns how many obstacles were left out for lack of slots.
   */
  size_t setObservedObstacles(const std::vector<ObservedObstacle>& observed, const vector_t& robotCom);

  static constexpr scalar_t kSlotGate = 1.0;           // [m] farthest an obstacle moves between two calls and keeps its slot
  static constexpr scalar_t kParkingDistance = 100.0;  // [m] from the CoM

  /** Obstacles in the tracking slots now. */
  int numActiveObstacles() const;

  /**
   * Obstacle block of the obstacle constraints' parameters for an OCP with `numSlots` slots (ObstacleP::DIM entries per
   * slot): the occupied tracking slots first, in slot order, then free ones (ObstacleP::ACTIVE 0: their rows constrain
   * nothing), parked. An OCP with fewer slots than obstacles gets the first ones.
   */
  vector_t getObstacleParameters(int numSlots) const;

  /** Id of the obstacle in each slot of an OCP with `numSlots` slots, laid out as getObstacleParameters(); -1 if free. */
  std::vector<int> getObstacleIds(int numSlots) const;

  /**
   * Trot parity at knot 0 of the horizon. Knot i stands on gaitPair(gaitOffset + i), so the horizon
   * alternates (FR, RL) / (FL, RR) starting from the parity of the current simulation step.
   */
  int getGaitOffset() const { return gaitOffset_; }
  void setGaitOffset(int gaitOffset) { gaitOffset_ = gaitOffset; }

  /**
   * Multiplier on v_obstacle * elapsedTime in the pessimistic keep-out radius.
   * 1.0 reproduces the CasADi reference; smaller values relax the worst-case disk on retry.
   */
  scalar_t getPessiScale() const { return pessiScale_; }
  void setPessiScale(scalar_t pessiScale) { pessiScale_ = pessiScale; }

 private:
  vector_t goal_;
  matrix_t obstacles_;
  vector_t obstacleRadii_;
  vector_t obstacleMaxSpeeds_;
  std::vector<ObstacleSlot> slots_;  // what each slot held at the last setObservedObstacles()
  int nextObstacleId_ = 0;           // id of the next new obstacle
  int gaitOffset_ = 0;
  scalar_t pessiScale_ = 1.0;
};

}  // namespace opti_pessi
