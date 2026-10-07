#include "opti_pessi_interface/OptiPessiMpc.h"

#include <algorithm>
#include <cmath>
#include <exception>

#include "opti_pessi_interface/definitions.h"
#include "opti_pessi_interface/initialization/OptiPessiInitializer.h"

namespace opti_pessi {

namespace {

/**
 * The measured yaw is wrapped to (-pi, pi]; the yaw along a solution is not. After a turn through +-pi every knot of the
 * shifted guess would sit 2 pi away from the state the solve starts from, and the solver gives up on it after a couple
 * of iterations. This moves the guess onto the measurement's branch.
 */
void alignYawBranch(ocs2::PrimalSolution& guess, const vector_t& robotState) {
  if (guess.stateTrajectory_.size() < 2) {
    return;
  }
  const scalar_t turn = 2.0 * M_PI;
  const scalar_t offset = turn * std::round((robotState(RobotX::TH) - guess.stateTrajectory_[1](RobotX::TH)) / turn);
  if (offset == 0.0) {
    return;
  }
  for (size_t k = 1; k < guess.stateTrajectory_.size(); ++k) {
    guess.stateTrajectory_[k](RobotX::TH) += offset;                // optimistic branch
    guess.stateTrajectory_[k](RobotX::DIM + RobotX::TH) += offset;  // pessimistic branch
  }
}

}  // namespace

OptiPessiMpc::OptiPessiMpc(ocs2::mpc::Settings mpcSettings, ocs2::ipm::Settings settings,
                           const ocs2::OptimalControlProblem& optimalControlProblem, const ocs2::Initializer& initializer,
                           std::shared_ptr<OptiPessiReferenceManager> referenceManagerPtr, OptiPessiModelParameters params)
    : MPC_BASE(std::move(mpcSettings)),
      solverPtr_(std::make_unique<ocs2::IpmSolver>(std::move(settings), optimalControlProblem, initializer)),
      referenceManagerPtr_(std::move(referenceManagerPtr)),
      params_(std::move(params)) {}

void OptiPessiMpc::reset() {
  MPC_BASE::reset();
  hasSolution_ = false;
  published_ = false;
  solverFailed_ = false;
}

bool OptiPessiMpc::run(scalar_t currentTime, const vector_t& currentState) {
  return MPC_BASE::run(currentTime, currentState) && published_;
}

bool OptiPessiMpc::solve(const vector_t& robotState, const ocs2::PrimalSolution* guess, ocs2::PrimalSolution& solution) {
  const vector_t initialState = packInitialState(robotState);
  const scalar_t finalTime = static_cast<scalar_t>(params_.N);
  try {
    // The guess sets only the primal trajectories: the solver warm-starts its slacks and duals from its own previous run.
    // A cold start, and a warm start after a failed run, clear them first.
    if (guess == nullptr || solverFailed_) {
      solverPtr_->reset();
    }
    if (guess != nullptr) {
      solverPtr_->run(0.0, initialState, finalTime, *guess);
    } else {
      solverPtr_->run(0.0, initialState, finalTime);
    }
    solution = solverPtr_->primalSolution(finalTime);
  } catch (const std::exception&) {
    solverFailed_ = true;
    return false;
  }

  // No check of the solution itself: only that there is one, complete and finite.
  const auto numKnots = static_cast<size_t>(params_.N);
  bool complete = solution.inputTrajectory_.size() >= numKnots && solution.stateTrajectory_.size() >= 2;
  for (size_t k = 0; complete && k < numKnots; ++k) {
    complete = solution.inputTrajectory_[k].allFinite();
  }
  solverFailed_ = !complete;
  return complete;
}

/**
 * One MPC step. The controller re-pushes the SAME phase until its clock advances, so this is called repeatedly on one
 * problem and the warm start is keyed on the gait offset, not on wall-clock time.
 */
void OptiPessiMpc::calculateController(scalar_t /*initTime*/, const vector_t& initState, scalar_t /*finalTime*/) {
  const int gaitOffset = referenceManagerPtr_->getGaitOffset();
  const vector_t robotState = extractRobotState(initState);

  // Warm start from the last solution: as it is on the same problem, shifted one knot on the next phase. The start state
  // is compared too because a walk restarts at phase 0 without this object being told. Anything else starts cold.
  const char* warmStart = "cold";
  ocs2::PrimalSolution guess;
  const ocs2::PrimalSolution* guessPtr = nullptr;
  if (!settings().coldStart_ && hasSolution_) {
    const bool sameProblem = gaitOffset == lastGaitOffset_ && lastStartState_.size() == robotState.size() &&
                             (lastStartState_ - robotState).cwiseAbs().maxCoeff() < 1e-9;
    if (sameProblem) {
      warmStart = "same phase";
      guessPtr = &lastSolution_;
    } else if (gaitOffset == lastGaitOffset_ + 1) {
      warmStart = "shifted";
      guess = shiftPrimalSolution(lastSolution_, robotState);
      alignYawBranch(guess, robotState);
      guessPtr = &guess;
    }
  }

  // The warm start, then a cold start if the solver failed on it. Nothing else.
  ocs2::PrimalSolution solution;
  bool solved = solve(robotState, guessPtr, solution);
  if (!solved && guessPtr != nullptr) {
    warmStart = "cold retry";
    solved = solve(robotState, nullptr, solution);
  }

  // Statistics first: a controller that reads the plan as soon as it is published then finds the statistics of its solve.
  SolveStatistics statistics;
  statistics.gaitOffset = gaitOffset;
  statistics.warmStart = warmStart;
  statistics.solved = solved;
  try {
    statistics.numIterations = solverPtr_->getIterationsLog().size();
    statistics.performance = solverPtr_->getPerformanceIndeces();
  } catch (const std::exception&) {
    // The solver failed before its first iteration: it has no log.
  }
  {
    std::lock_guard<std::mutex> lock(statisticsMutex_);
    lastSolveStatistics_ = statistics;
  }

  published_ = solved;
  if (!solved) {
    return;  // the solver gave nothing: the previous plan stays
  }

  Plan plan;
  plan.phase = static_cast<size_t>(std::max(gaitOffset, 0));
  plan.startState = robotState;
  plan.source = warmStart;
  const auto numKnots = static_cast<size_t>(params_.N);
  for (size_t k = 0; k < numKnots; ++k) {
    plan.inputs.push_back(extractRobotInput(solution.inputTrajectory_[k]));
  }
  lastSolution_ = std::move(solution);
  lastGaitOffset_ = gaitOffset;
  lastStartState_ = robotState;
  hasSolution_ = true;
  {
    std::lock_guard<std::mutex> lock(planMutex_);
    plan.sequence = planSequence_ + 1;
    latestPlan_ = std::move(plan);
    planSequence_ = latestPlan_.sequence;
  }
}

OptiPessiMpc::Plan OptiPessiMpc::getLatestPlan() const {
  std::lock_guard<std::mutex> lock(planMutex_);
  return latestPlan_;
}

OptiPessiMpc::SolveStatistics OptiPessiMpc::getLastSolveStatistics() const {
  std::lock_guard<std::mutex> lock(statisticsMutex_);
  return lastSolveStatistics_;
}

}  // namespace opti_pessi
