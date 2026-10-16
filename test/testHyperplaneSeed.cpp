#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "opti_pessi_interface/OptiPessiReferenceManager.h"
#include "opti_pessi_interface/initialization/OptiPessiInitializer.h"

using namespace opti_pessi;

namespace {

OptiPessiModelParameters aliengo(int numSlots) {
  OptiPessiModelParameters params;
  params.hipOffsets = {(vector_t(2) << 0.2407, 0.134).finished(), (vector_t(2) << 0.2407, -0.134).finished(),
                       (vector_t(2) << -0.2407, 0.134).finished(), (vector_t(2) << -0.2407, -0.134).finished()};
  params.goal = vector_t::Zero(2);
  params.numSlots = numSlots;
  return params;
}

/** Slack of the most advanced hip behind the seeded mid-step plane of `slot`, and of the obstacle's disk beyond it. */
struct Slacks {
  scalar_t hips;
  scalar_t obstacle;
};

Slacks seededSlacks(const OptiPessiModelParameters& params, const OptiPessiReferenceManager& referenceManager, const vector_t& robotState,
                    int slot) {
  vector_t input = vector_t::Zero(augInputDim(params.numObstacles()));
  seedHyperplanes(input, optiHyperplaneOffset(), params, robotState, referenceManager);
  const int base = optiHyperplaneOffset() + kHyperplaneVarsPerObs * slot;
  const scalar_t phi = input(base + Hyperplane::MID_PHI);
  const scalar_t b = input(base + Hyperplane::MID_B);
  const vector_t a = (vector_t(2) << std::cos(phi), std::sin(phi)).finished();

  Slacks slacks{1e9, 0.0};
  for (const vector_t& hip : params.hipOffsets) {
    slacks.hips = std::min(slacks.hips, -(a.dot(robotState.head(2) + hip) + b));
  }
  const vector_t p = referenceManager.getObstacleParameters(params.numObstacles());
  const vector_t o = (vector_t(2) << p(ObstacleP::DIM * slot + ObstacleP::X), p(ObstacleP::DIM * slot + ObstacleP::Y)).finished();
  slacks.obstacle = a.dot(o) + b - p(ObstacleP::DIM * slot + ObstacleP::RADIUS);
  return slacks;
}

}  // namespace

// A plane seeded against the hips starts every hip row at zero slack: the barrier then pushes the robot away from the
// plane, and for a parked slot 100 m off the plane has tens of metres to travel before the solve can settle.
TEST(HyperplaneSeed, ThePlaneStartsHalfwayBetweenTheHipsAndTheKeepOutDisk) {
  const auto params = aliengo(2);
  OptiPessiReferenceManager referenceManager(params);
  const vector_t robotState = vector_t::Zero(RobotX::DIM);
  referenceManager.setObservedObstacles({ObservedObstacle{1.5, 0.5, 0.3, 1.0}}, robotState.head(2));  // slot 1 parked

  for (int slot = 0; slot < 2; ++slot) {
    const Slacks slacks = seededSlacks(params, referenceManager, robotState, slot);
    EXPECT_GT(slacks.hips, 0.1) << "slot " << slot;
    EXPECT_NEAR(slacks.hips, slacks.obstacle, 1e-9) << "slot " << slot;
  }
}

TEST(HyperplaneSeed, AnObstacleInsideTheHullGetsThePlaneAtTheHips) {
  const auto params = aliengo(1);
  OptiPessiReferenceManager referenceManager(params);
  const vector_t robotState = vector_t::Zero(RobotX::DIM);
  referenceManager.setObservedObstacles({ObservedObstacle{0.2, 0.0, 0.3, 1.0}}, robotState.head(2));

  const Slacks slacks = seededSlacks(params, referenceManager, robotState, 0);
  EXPECT_NEAR(slacks.hips, 0.0, 1e-9);
}

namespace {

/** A two-knot solution of the OCP with `numSlots` slots, every hyperplane variable of slot j and branch b set to 10 j + b. */
ocs2::PrimalSolution markedSolution(int numSlots) {
  ocs2::PrimalSolution solution;
  for (int k = 0; k < 2; ++k) {
    solution.timeTrajectory_.push_back(k);
    solution.stateTrajectory_.push_back(vector_t::Zero(AUG_STATE_DIM));
    vector_t u = vector_t::Constant(augInputDim(numSlots), -1.0);
    for (int j = 0; j < numSlots; ++j) {
      u.segment(optiHyperplaneOffset() + kHyperplaneVarsPerObs * j, kHyperplaneVarsPerObs).setConstant(10.0 * j);
      u.segment(pessiHyperplaneOffset(numSlots) + kHyperplaneVarsPerObs * j, kHyperplaneVarsPerObs).setConstant(10.0 * j + 1.0);
    }
    solution.inputTrajectory_.push_back(u);
  }
  solution.stateTrajectory_.push_back(vector_t::Zero(AUG_STATE_DIM));
  return solution;
}

}  // namespace

TEST(RemapObstacleSlots, AnObstacleKeepsItsPlanesInItsNewSlotAndANewOneIsSeeded) {
  auto params = aliengo(3);
  OptiPessiReferenceManager referenceManager(params);
  referenceManager.setObservedObstacles({ObservedObstacle{2.0, 0.0, 0.2, 1.0}, ObservedObstacle{0.0, 3.0, 0.2, 1.0}}, vector_t::Zero(2));
  params.numSlots = 2;

  // Solved with obstacles 7 and 4 in its 2 slots; now obstacle 4 is in slot 0 and a new one, 9, in slot 1.
  const ocs2::PrimalSolution remapped = remapObstacleSlots(markedSolution(2), {7, 4}, {4, 9}, params, referenceManager);
  ASSERT_EQ(remapped.inputTrajectory_.size(), 2u);
  for (const vector_t& u : remapped.inputTrajectory_) {
    ASSERT_EQ(u.size(), augInputDim(2));
    EXPECT_TRUE((u.head(2 * RobotU::DIM).array() == -1.0).all());  // the robot inputs as they were
    // Obstacle 4 was in slot 1: its planes, both branches.
    EXPECT_TRUE((u.segment(optiHyperplaneOffset(), kHyperplaneVarsPerObs).array() == 10.0).all());
    EXPECT_TRUE((u.segment(pessiHyperplaneOffset(2), kHyperplaneVarsPerObs).array() == 11.0).all());
    // Obstacle 9 is new: seeded towards the obstacle in OCP slot 1, (0, 3), not copied.
    const scalar_t phi = u(optiHyperplaneOffset() + kHyperplaneVarsPerObs + Hyperplane::MID_PHI);
    EXPECT_NEAR(phi, M_PI / 2, 1e-9);
  }
  EXPECT_EQ(remapped.stateTrajectory_.size(), 3u);
}

TEST(RemapObstacleSlots, ToASmallerOcp) {
  auto params = aliengo(3);
  OptiPessiReferenceManager referenceManager(params);
  referenceManager.setObservedObstacles({ObservedObstacle{2.0, 0.0, 0.2, 1.0}}, vector_t::Zero(2));
  params.numSlots = 1;

  const ocs2::PrimalSolution remapped = remapObstacleSlots(markedSolution(3), {1, 2, 3}, {3}, params, referenceManager);
  for (const vector_t& u : remapped.inputTrajectory_) {
    ASSERT_EQ(u.size(), augInputDim(1));
    EXPECT_TRUE((u.segment(optiHyperplaneOffset(), kHyperplaneVarsPerObs).array() == 20.0).all());
    EXPECT_TRUE((u.segment(pessiHyperplaneOffset(1), kHyperplaneVarsPerObs).array() == 21.0).all());
  }
}
