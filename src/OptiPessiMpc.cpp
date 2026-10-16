#include "opti_pessi_interface/OptiPessiMpc.h"

#include <algorithm>
#include <chrono>
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

OptiPessiMpc::OptiPessiMpc(const OptiPessiInterface& interface)
    : MPC_BASE(interface.mpcSettings()),
      interface_(interface),
      referenceManagerPtr_(interface.getOptiPessiReferenceManagerPtr()),
      params_(interface.modelParameters()),
      iterationCap_(interface.ipmSettings().ipmIteration) {
  for (int numSlots = 1; numSlots <= interface.numObstacles(); ++numSlots) {
    solvers_.push_back(std::make_unique<ocs2::IpmSolver>(interface.ipmSettings(), interface.getOptimalControlProblem(numSlots),
                                                          interface.getInitializer(numSlots)));
  }
  activeSolver_ = solvers_.size() - 1;
}

void OptiPessiMpc::setReferenceManager(std::shared_ptr<ocs2::ReferenceManagerInterface> referenceManagerPtr) {
  for (auto& solver : solvers_) {
    solver->setReferenceManager(referenceManagerPtr);
  }
}

void OptiPessiMpc::reset() {
  MPC_BASE::reset();
  for (auto& solver : solvers_) {
    solver->reset();
  }
  hasSolution_ = false;
  lastObstacleIds_.clear();
  published_ = false;
  solverFailed_ = false;
}

bool OptiPessiMpc::run(scalar_t currentTime, const vector_t& currentState) {
  return MPC_BASE::run(currentTime, currentState) && published_;
}

bool OptiPessiMpc::solve(const vector_t& robotState, const ocs2::PrimalSolution* guess, bool freshDuals,
                         ocs2::PrimalSolution& solution) {
  const vector_t initialState = packInitialState(robotState);
  const scalar_t finalTime = static_cast<scalar_t>(params_.N);
  ocs2::IpmSolver& solver = *solvers_[activeSolver_];
  try {
    // The guess sets only the primal trajectories: the solver warm-starts its slacks and duals from its own previous run.
    // A cold start, a warm start after a failed run and one for other obstacles clear them first.
    if (guess == nullptr || solverFailed_ || freshDuals) {
      solver.reset();
    }
    if (guess != nullptr) {
      solver.run(0.0, initialState, finalTime, *guess);
    } else {
      solver.run(0.0, initialState, finalTime);
    }
    solution = solver.primalSolution(finalTime);
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

  // The OCP with one slot per obstacle present, and the obstacle each of its slots holds.
  const int numSlots = interface_.ocpSlots(referenceManagerPtr_->numActiveObstacles());
  const std::vector<int> obstacleIds = referenceManagerPtr_->getObstacleIds(numSlots);
  activeSolver_ = static_cast<size_t>(numSlots - 1);

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
  const bool samePhase = guessPtr == &lastSolution_;

  // Other obstacles in the slots than in the warm start (one more, one less, one replaced): each remaining obstacle keeps
  // its planes, a new one gets them seeded, and the solver of this size starts its slacks and duals afresh.
  const bool obstaclesChanged = guessPtr != nullptr && obstacleIds != lastObstacleIds_;
  if (obstaclesChanged) {
    guess = remapObstacleSlots(*guessPtr, lastObstacleIds_, obstacleIds, interface_.modelParameters(numSlots), *referenceManagerPtr_);
    guessPtr = &guess;
  }

  // The warm start, then a cold start if the solver failed on it. Nothing else.
  const auto solveStart = std::chrono::steady_clock::now();
  ocs2::PrimalSolution solution;
  bool solved = solve(robotState, guessPtr, obstaclesChanged, solution);
  const bool coldRetry = !solved && guessPtr != nullptr;
  if (coldRetry) {
    warmStart = "cold retry";
    solved = solve(robotState, nullptr, true, solution);
  }
  const scalar_t solveTime = std::chrono::duration<scalar_t, std::milli>(std::chrono::steady_clock::now() - solveStart).count();

  // Statistics first: a controller that reads the plan as soon as it is published then finds the statistics of its solve.
  ocs2::IpmSolver& solver = *solvers_[activeSolver_];
  SolveStatistics statistics;
  statistics.gaitOffset = gaitOffset;
  statistics.warmStart = warmStart;
  statistics.obstacleSlots = numSlots;
  statistics.obstaclesChanged = obstaclesChanged;
  statistics.solved = solved;
  try {
    statistics.numIterations = solver.getIterationsLog().size();
    statistics.performance = solver.getPerformanceIndeces();
  } catch (const std::exception&) {
    // The solver failed before its first iteration: it has no log.
  }
  // The profile is read here, on the thread that runs the solver, so no other thread touches the solver's timers.
  std::string solverProfile = solver.getBenchmarkingInfo();
  {
    std::lock_guard<std::mutex> lock(statisticsMutex_);
    lastSolveStatistics_ = statistics;
    ++totals_.solves;
    totals_.samePhase += samePhase ? 1 : 0;
    totals_.shifted += guessPtr != nullptr && !samePhase ? 1 : 0;
    totals_.cold += guessPtr == nullptr ? 1 : 0;
    totals_.coldRetries += coldRetry ? 1 : 0;
    totals_.obstacleChanges += obstaclesChanged ? 1 : 0;
    totals_.unsolved += solved ? 0 : 1;
    totals_.iterations += statistics.numIterations;
    totals_.atIterationCap += statistics.numIterations >= iterationCap_ ? 1 : 0;
    totals_.solveTimeSum += solveTime;
    totals_.solveTimeMax = std::max(totals_.solveTimeMax, solveTime);
    totals_.solverProfile = std::move(solverProfile);
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
  lastObstacleIds_ = obstacleIds;
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

OptiPessiMpc::SolveTotals OptiPessiMpc::getSolveTotals() const {
  std::lock_guard<std::mutex> lock(statisticsMutex_);
  return totals_;
}

}  // namespace opti_pessi
