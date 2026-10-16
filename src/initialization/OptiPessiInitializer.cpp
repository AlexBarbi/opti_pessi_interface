#include "opti_pessi_interface/initialization/OptiPessiInitializer.h"

#include <algorithm>
#include <cmath>

#include "opti_pessi_interface/LipKinematics.h"

namespace opti_pessi {

vector_t packInitialState(const vector_t& robotState) {
  vector_t x = vector_t::Zero(AUG_STATE_DIM);
  x.segment(0, RobotX::DIM) = robotState;
  x.segment(RobotX::DIM, RobotX::DIM) = robotState;
  return x;  // clock starts at zero
}

vector_t extractRobotState(const vector_t& augmentedState) {
  return augmentedState.head(RobotX::DIM);
}

vector_t extractRobotInput(const vector_t& augmentedInput) {
  return augmentedInput.head(RobotU::DIM);
}

vector_t defaultRobotInput(const OptiPessiModelParameters& params, const vector_t& robotState, int knotParity) {
  vector_t u = vector_t::Zero(RobotU::DIM);
  const auto next = gaitPair(knotParity + 1);
  const vector_t c = robotState.head(2);
  const vector_t dc = robotState.segment(RobotX::DCX, 2);
  const scalar_t theta = robotState(RobotX::TH);
  const scalar_t w = params.omega();
  const scalar_t dt = params.dtCost0;

  // Project the CoM forward with the LIP velocity term before placing the footholds.
  const vector_t cNextApprox = c + (std::sinh(w * dt) / w) * dc;
  u.segment(RobotU::P0X, 2) = cNextApprox + applyR(theta, hipOf(params, next[0]));
  u.segment(RobotU::P1X, 2) = cNextApprox + applyR(theta, hipOf(params, next[1]));
  u(RobotU::ALPHA) = 0.5;
  u(RobotU::DT) = params.dtCost0;
  u(RobotU::BETA) = 0.5;
  u(RobotU::GAMMA) = 0.5;
  return u;
}

void seedHyperplane(vector_t& input, int hyperplaneOffset, int slot, const OptiPessiModelParameters& params, const vector_t& robotState,
                    const vector_t& obstacleParameters) {
  const vector_t c = robotState.head(2);
  const scalar_t theta = robotState(RobotX::TH);
  const int obstacle = ObstacleP::DIM * slot;
  vector_t o(2);
  o << obstacleParameters(obstacle + ObstacleP::X), obstacleParameters(obstacle + ObstacleP::Y);

  // Normal points from the robot towards the obstacle.
  vector_t a = o - c;
  const scalar_t norm = a.norm();
  if (norm < 1e-9) {
    a << 1.0, 0.0;
  } else {
    a /= norm;
  }

  // Offset halfway between the most advanced robot point and the obstacle's disk, so a^T y + b <= 0 holds for all hips
  // and every row starts with room on its side. A plane against the hips starts their rows at zero slack: the barrier
  // then pushes the robot back from it (the first step went backwards at -0.5 m/s from rest), and for a slot parked
  // 100 m off the plane has tens of metres to travel before a 50-iteration solve settles.
  scalar_t maxProjection = a.dot(c);
  for (int f = 0; f < 4; ++f) {
    const vector_t hipWorld = c + applyR(theta, hipOf(params, static_cast<Foot>(f)));
    maxProjection = std::max(maxProjection, a.dot(hipWorld));
  }
  const scalar_t gap = std::max(scalar_t(0), a.dot(o) - obstacleParameters(obstacle + ObstacleP::RADIUS) - maxProjection);
  const scalar_t b = -(maxProjection + 0.5 * gap);

  // Both planes get the same seed: at the initial guess the mid-step and landing poses coincide,
  // and the solver separates them from there.
  const int base = hyperplaneOffset + kHyperplaneVarsPerObs * slot;
  const scalar_t phi = std::atan2(a(1), a(0));
  input(base + Hyperplane::MID_PHI) = phi;
  input(base + Hyperplane::MID_B) = b;
  input(base + Hyperplane::LAND_PHI) = phi;
  input(base + Hyperplane::LAND_B) = b;
}

void seedHyperplanes(vector_t& input, int hyperplaneOffset, const OptiPessiModelParameters& params, const vector_t& robotState,
                     const OptiPessiReferenceManager& referenceManager) {
  const vector_t obstacleParameters = referenceManager.getObstacleParameters(params.numObstacles());
  for (int j = 0; j < params.numObstacles(); ++j) {
    seedHyperplane(input, hyperplaneOffset, j, params, robotState, obstacleParameters);
  }
}

ocs2::PrimalSolution remapObstacleSlots(const ocs2::PrimalSolution& solution, const std::vector<int>& fromIds,
                                        const std::vector<int>& toIds, const OptiPessiModelParameters& toParams,
                                        const OptiPessiReferenceManager& referenceManager) {
  const int fromSlots = static_cast<int>(fromIds.size());
  const int toSlots = toParams.numObstacles();
  const vector_t obstacleParameters = referenceManager.getObstacleParameters(toSlots);

  ocs2::PrimalSolution out = solution;
  for (size_t k = 0; k < solution.inputTrajectory_.size(); ++k) {
    const vector_t& from = solution.inputTrajectory_[k];
    vector_t to = vector_t::Zero(augInputDim(toSlots));
    to.head(2 * RobotU::DIM) = from.head(2 * RobotU::DIM);
    // The branches' states at this knot, where a new obstacle's planes are seeded from.
    const vector_t& state = solution.stateTrajectory_[std::min(k, solution.stateTrajectory_.size() - 1)];
    for (int j = 0; j < toSlots; ++j) {
      const auto previous = std::find(fromIds.begin(), fromIds.end(), toIds[static_cast<size_t>(j)]);
      if (toIds[static_cast<size_t>(j)] >= 0 && previous != fromIds.end()) {
        const int i = static_cast<int>(previous - fromIds.begin());
        to.segment(optiHyperplaneOffset() + kHyperplaneVarsPerObs * j, kHyperplaneVarsPerObs) =
            from.segment(optiHyperplaneOffset() + kHyperplaneVarsPerObs * i, kHyperplaneVarsPerObs);
        to.segment(pessiHyperplaneOffset(toSlots) + kHyperplaneVarsPerObs * j, kHyperplaneVarsPerObs) =
            from.segment(pessiHyperplaneOffset(fromSlots) + kHyperplaneVarsPerObs * i, kHyperplaneVarsPerObs);
      } else {
        seedHyperplane(to, optiHyperplaneOffset(), j, toParams, state.segment(0, RobotX::DIM), obstacleParameters);
        seedHyperplane(to, pessiHyperplaneOffset(toSlots), j, toParams, state.segment(RobotX::DIM, RobotX::DIM), obstacleParameters);
      }
    }
    out.inputTrajectory_[k] = std::move(to);
  }
  return out;
}

vector_t augmentedLipStep(const OptiPessiModelParameters& params, const vector_t& augmentedState, const vector_t& augmentedInput) {
  const scalar_t w = params.omega();
  vector_t xNext = augmentedState;
  xNext.segment(0, RobotX::DIM) =
      lipMapScalar(augmentedState.head(RobotX::DIM), augmentedInput.head(RobotU::DIM), w, params.mass, params.inertia);
  xNext.segment(RobotX::DIM, RobotX::DIM) = lipMapScalar(augmentedState.segment(RobotX::DIM, RobotX::DIM),
                                                         augmentedInput.segment(RobotU::DIM, RobotU::DIM), w, params.mass,
                                                         params.inertia);
  xNext(CLOCK_INDEX) = augmentedState(CLOCK_INDEX) + augmentedInput(RobotU::DIM + RobotU::DT);
  return xNext;
}

namespace {

/** Fills one augmented input with the nominal inputs and seeded hyperplanes of both branches. */
vector_t nominalAugmentedInput(const OptiPessiModelParameters& params, const OptiPessiReferenceManager& referenceManager,
                               const vector_t& augmentedState, int knotParity) {
  const int numObstacles = params.numObstacles();
  vector_t u = vector_t::Zero(augInputDim(numObstacles));

  const vector_t xOpti = augmentedState.segment(0, RobotX::DIM);
  const vector_t xPessi = augmentedState.segment(RobotX::DIM, RobotX::DIM);

  u.segment(0, RobotU::DIM) = defaultRobotInput(params, xOpti, knotParity);
  u.segment(RobotU::DIM, RobotU::DIM) = defaultRobotInput(params, xPessi, knotParity);
  seedHyperplanes(u, optiHyperplaneOffset(), params, xOpti, referenceManager);
  seedHyperplanes(u, pessiHyperplaneOffset(numObstacles), params, xPessi, referenceManager);
  return u;
}

}  // namespace

OptiPessiInitializer::OptiPessiInitializer(OptiPessiModelParameters params, const OptiPessiReferenceManager& referenceManager)
    : params_(std::move(params)), referenceManagerPtr_(&referenceManager) {}

void OptiPessiInitializer::compute(scalar_t time, const vector_t& state, scalar_t /*nextTime*/, vector_t& input,
                                   vector_t& nextState) {
  const int i = intervalIndex(time, params_.N);
  const int knotParity = referenceManagerPtr_->getGaitOffset() + i;
  input = nominalAugmentedInput(params_, *referenceManagerPtr_, state, knotParity);
  nextState = augmentedLipStep(params_, state, input);
}

ocs2::PrimalSolution shiftPrimalSolution(const ocs2::PrimalSolution& solution, const vector_t& robotState) {
  ocs2::PrimalSolution out = solution;
  const int numStates = static_cast<int>(solution.stateTrajectory_.size());
  const int numInputs = static_cast<int>(solution.inputTrajectory_.size());
  if (numStates < 2 || numInputs < 1) {
    return out;
  }

  // Shift the inputs (hyperplane variables included, they live in the input) one knot forward,
  // repeating the last one.
  for (int i = 0; i < numInputs - 1; ++i) {
    out.inputTrajectory_[static_cast<size_t>(i)] = solution.inputTrajectory_[static_cast<size_t>(i + 1)];
  }
  out.inputTrajectory_[static_cast<size_t>(numInputs - 1)] = solution.inputTrajectory_[static_cast<size_t>(numInputs - 1)];

  // Shift the STORED states rather than re-integrating them.
  //
  // Re-integrating (x_{i+1} = lipMap(x_i, u_i) from the new measured x_0) looks more principled --
  // it makes the guess satisfy the dynamics exactly -- but it is wrong here: the LIP is
  // exponentially unstable, |dc_{i+1}/dc_i| = cosh(omega*dt) ~ 1.6-1.9 over the configured dt range,
  // and the footholds in u are ABSOLUTE world positions. Any mismatch between the measured state and
  // the previous plan is therefore amplified at every knot, compounding along the horizon and again
  // at every MPC step. The solver then cannot repair the guess and simply returns it, and the closed
  // loop diverges.
  //
  // The reference implementation shifts the stored trajectory (ocp_quadruped.py:455-457,
  // `set_initial(x[:, i], x_guess[:, i + 1])`), which stays near the previous solution. Only the
  // first knot is replaced by the measurement; the solver closes the resulting defects itself.
  for (int i = 0; i < numStates - 1; ++i) {
    out.stateTrajectory_[static_cast<size_t>(i)] = solution.stateTrajectory_[static_cast<size_t>(i + 1)];
  }
  out.stateTrajectory_[static_cast<size_t>(numStates - 1)] = solution.stateTrajectory_[static_cast<size_t>(numStates - 1)];
  out.stateTrajectory_[0] = packInitialState(robotState);
  return out;
}

}  // namespace opti_pessi
