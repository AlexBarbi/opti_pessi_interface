// A free obstacle slot must constrain nothing: its rows are the constant 1, with no derivative, in the stage and in the
// terminal obstacle constraint. A far "parked" obstacle instead pushed the robot away from its parking spot through the
// barrier on the hip rows whenever the solve stopped short of convergence.
#include <string>

#include <gtest/gtest.h>

#include <ocs2_core/PreComputation.h>

#include "opti_pessi_interface/LipKinematics.h"
#include "opti_pessi_interface/OptiPessiReferenceManager.h"
#include "opti_pessi_interface/constraint/StageInequalityConstraint.h"
#include "opti_pessi_interface/constraint/TerminalObstacleConstraint.h"
#include "opti_pessi_interface/initialization/OptiPessiInitializer.h"

using namespace opti_pessi;

namespace {

constexpr int kSlots = 2;
constexpr int kPathRowsPerBranch = 11;
constexpr int kCollisionRowsPerObstacle = 12;
const std::string kLibraryFolder = "/tmp/ocs2/opti_pessi_interface_test_free_slots";

OptiPessiModelParameters aliengo() {
  OptiPessiModelParameters params;
  params.hipOffsets = {(vector_t(2) << 0.2407, 0.134).finished(), (vector_t(2) << 0.2407, -0.134).finished(),
                       (vector_t(2) << -0.2407, 0.134).finished(), (vector_t(2) << -0.2407, -0.134).finished()};
  params.goal = (vector_t(2) << 2.0, 0.0).finished();
  params.numSlots = kSlots;
  return params;
}

/** Robot a little off the origin, walking, with its feet under the hips: nothing degenerate in the rows. */
vector_t augmentedState(const OptiPessiModelParameters& params) {
  vector_t x = vector_t::Zero(RobotX::DIM);
  x(RobotX::CX) = 0.1;
  x(RobotX::CY) = -0.05;
  x(RobotX::TH) = 0.2;
  x(RobotX::DCX) = 0.3;
  x.segment<2>(RobotX::P0X) = x.head<2>() + applyR(x(RobotX::TH), hipOf(params, Foot::FR));
  x.segment<2>(RobotX::P1X) = x.head<2>() + applyR(x(RobotX::TH), hipOf(params, Foot::RL));
  vector_t augmented = packInitialState(x);
  augmented(CLOCK_INDEX) = 0.5;
  return augmented;
}

vector_t augmentedInput(const OptiPessiModelParameters& params, const OptiPessiReferenceManager& referenceManager,
                        const vector_t& state) {
  vector_t u = vector_t::Zero(params.inputDim());
  u.head(RobotU::DIM) = defaultRobotInput(params, extractRobotState(state), 0);
  u.segment(RobotU::DIM, RobotU::DIM) = u.head(RobotU::DIM);
  seedHyperplanes(u, optiHyperplaneOffset(), params, extractRobotState(state), referenceManager);
  seedHyperplanes(u, pessiHyperplaneOffset(kSlots), params, extractRobotState(state), referenceManager);
  return u;
}

}  // namespace

TEST(FreeSlots, StageRowsOfAFreeSlotAreOneWithNoDerivative) {
  const auto params = aliengo();
  OptiPessiReferenceManager referenceManager(params);
  const vector_t state = augmentedState(params);
  referenceManager.setObservedObstacles({ObservedObstacle{1.5, 0.4, 0.2, 1.0}}, extractRobotState(state).head(2));  // slot 1 free

  StageInequalityConstraint constraint(params, referenceManager, kLibraryFolder, true);
  const vector_t input = augmentedInput(params, referenceManager, state);
  ocs2::PreComputation preComputation;
  const auto approximation = constraint.getLinearApproximation(0.0, state, input, preComputation);

  const int rowsPerBranch = kPathRowsPerBranch + kCollisionRowsPerObstacle * kSlots;
  for (int branch = 0; branch < 2; ++branch) {
    const int taken = branch * rowsPerBranch + kPathRowsPerBranch;
    const int free = taken + kCollisionRowsPerObstacle;
    for (int row = free; row < free + kCollisionRowsPerObstacle; ++row) {
      EXPECT_DOUBLE_EQ(approximation.f(row), 1.0) << "branch " << branch << " row " << row;
      EXPECT_EQ(approximation.dfdx.row(row).norm(), 0.0) << "branch " << branch << " row " << row;
      EXPECT_EQ(approximation.dfdu.row(row).norm(), 0.0) << "branch " << branch << " row " << row;
    }
    // The slot with an obstacle still constrains the robot.
    EXPECT_GT(approximation.dfdx.middleRows(taken, kCollisionRowsPerObstacle).norm(), 0.0) << "branch " << branch;
  }
}

TEST(FreeSlots, TerminalRowOfAFreeSlotIsOneWithNoDerivative) {
  const auto params = aliengo();
  OptiPessiReferenceManager referenceManager(params);
  const vector_t state = augmentedState(params);
  referenceManager.setObservedObstacles({ObservedObstacle{1.5, 0.4, 0.2, 1.0}}, extractRobotState(state).head(2));  // slot 1 free

  TerminalObstacleConstraint constraint(params, referenceManager, kLibraryFolder, true);
  ocs2::PreComputation preComputation;
  const auto approximation = constraint.getLinearApproximation(0.0, state, preComputation);

  for (int branch = 0; branch < 2; ++branch) {
    const int taken = branch * kSlots;
    const int free = taken + 1;
    EXPECT_DOUBLE_EQ(approximation.f(free), 1.0) << "branch " << branch;
    EXPECT_EQ(approximation.dfdx.row(free).norm(), 0.0) << "branch " << branch;
    EXPECT_GT(approximation.dfdx.row(taken).norm(), 0.0) << "branch " << branch;
  }
}
