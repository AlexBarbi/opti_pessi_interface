#include "opti_pessi_interface/OptiPessiMpc.h"

#include <algorithm>

#include "opti_pessi_interface/LipKinematics.h"
#include "opti_pessi_interface/OptiPessiInterface.h"
#include "opti_pessi_interface/initialization/OptiPessiInitializer.h"

namespace opti_pessi {

OptiPessiMpc::OptiPessiMpc(ocs2::mpc::Settings mpcSettings, ocs2::ipm::Settings settings,
                           const ocs2::OptimalControlProblem& optimalControlProblem, const ocs2::Initializer& initializer,
                           std::shared_ptr<OptiPessiReferenceManager> referenceManagerPtr, OptiPessiModelParameters params)
    : MPC_BASE(std::move(mpcSettings)),
      solverPtr_(std::make_unique<ocs2::IpmSolver>(std::move(settings), optimalControlProblem, initializer)),
      referenceManagerPtr_(std::move(referenceManagerPtr)),
      params_(std::move(params)),
      evaluationProblemPtr_(std::make_unique<ocs2::OptimalControlProblem>(optimalControlProblem)) {}

void OptiPessiMpc::reset() {
  MPC_BASE::reset();
  hasSolution_ = false;
  nominalPublished_ = false;
  shiftRejected_ = false;
  solverStateRejected_ = false;
}

bool OptiPessiMpc::run(scalar_t currentTime, const vector_t& currentState) {
  return MPC_BASE::run(currentTime, currentState) && nominalPublished_;
}

/**
 * One MPC step. The controller re-pushes the SAME phase until its clock advances, so this is called
 * repeatedly on one problem and the warm start is keyed on the gait offset, not on wall-clock time.
 *
 * A solve is judged on the step it applies (evaluateSolve(): SolveOutcome::ok), not on its prediction. One whose first
 * step is usable but whose plan does not start from the state that step leads to (!dynamicsConsistent()) is published as
 * "first step" when its phase has no plan yet: knot 0 as solved, the later knots checked against the real successor (see
 * below). It never becomes a warm start: warm-starting from such a solve left the solver stuck on its own failed output
 * (fewer iterations every solve, same violated plan) until the robot fell.
 *
 * A solve whose first step is rejected is not executed: while its phase has no plan it is published as "failed", without
 * inputs, and the controller stops the robot in stance and restarts the MPC cold. A phase that has a plan keeps it unless
 * the new solve is accepted and consistent.
 */
void OptiPessiMpc::calculateController(scalar_t /*initTime*/, const vector_t& initState, scalar_t /*finalTime*/) {
  const int gaitOffset = referenceManagerPtr_->getGaitOffset();
  const vector_t robotState = extractRobotState(initState);

  // The start state is compared too because a restart from stance begins at phase 0 again without resetting the MPC.
  const auto forThisProblem = [&](int phase, const vector_t& startState) {
    return gaitOffset == phase && startState.size() == robotState.size() && (startState - robotState).cwiseAbs().maxCoeff() < 1e-9;
  };
  // lastSolution_ was accepted for this very problem: the warm start as it is.
  const bool samePhase = hasSolution_ && forThisProblem(lastGaitOffset_, lastStartState_);
  // The phase already has a published plan (only this thread writes latestPlan_).
  const bool phaseHasPlan =
      planSequence_ > 0 && !latestPlan_.failed && forThisProblem(static_cast<int>(latestPlan_.phase), latestPlan_.startState);

  // Warm start from the last accepted solution: as-is on the same phase, shifted one knot on the next phase unless that
  // shifted guess was already rejected there. Anything else starts cold.
  const char* warmStart = "cold";
  ocs2::PrimalSolution guess;
  const ocs2::PrimalSolution* guessPtr = nullptr;
  bool shifted = false;
  if (!settings().coldStart_ && hasSolution_) {
    if (samePhase) {
      warmStart = "same phase";
      guessPtr = &lastSolution_;
    } else if (gaitOffset == lastGaitOffset_ + 1 && !shiftRejected_) {
      warmStart = "shifted";
      guess = shiftPrimalSolution(lastSolution_, robotState);
      guessPtr = &guess;
      shifted = true;
    }
  }
  // The guess sets only the primal trajectories: the solver warm-starts its slacks and duals from its own previous run.
  // After a rejected run those are cleared, so they are initialized again around the guess.
  if (guessPtr != nullptr && solverStateRejected_) {
    solverPtr_->reset();
  }

  // A single solve (realTimeIteration skips the cold retry and the keep-out continuation).
  scalar_t acceptedScale = 1.0;
  SolveOutcome outcome = solveWithRetries(*solverPtr_, *evaluationProblemPtr_, params_, *referenceManagerPtr_, robotState, guessPtr,
                                          /*realTimeIteration=*/true, /*verbose=*/false, acceptedScale);

  // A phase with no plan yet stops the robot if its solve is rejected, so a warm-started solve that gave no trustworthy
  // plan gets one cold retry first: a shifted guess the measured state has drifted away from can leave the solver stuck
  // after one or two iterations. Re-solves of a phase that has a plan stay single attempts.
  const bool guessWasShifted = shifted;
  bool coldRetry = false;
  if (guessPtr != nullptr && !phaseHasPlan && !outcome.planTrustworthy()) {
    const SolveOutcome cold = solveWithRetries(*solverPtr_, *evaluationProblemPtr_, params_, *referenceManagerPtr_, robotState, nullptr,
                                               /*realTimeIteration=*/true, /*verbose=*/false, acceptedScale);
    coldRetry = true;
    if (cold.planTrustworthy() || (!outcome.ok && cold.ok)) {
      outcome = cold;
      warmStart = "cold retry";
    }
  }

  const auto numKnots = static_cast<size_t>(params_.N);
  const bool complete = outcome.solution.inputTrajectory_.size() >= numKnots;
  // accepted: the applied step is usable. reusable: the plan also starts from the state that step leads to, so it is a
  // prediction too, and the next warm start.
  const bool accepted = complete && outcome.ok;
  const bool reusable = accepted && outcome.dynamicsConsistent();
  // After a cold retry the solver holds the retry's slacks and duals, whichever outcome was kept.
  solverStateRejected_ = !reusable || coldRetry;
  if (reusable) {
    lastSolution_ = outcome.solution;
    lastGaitOffset_ = gaitOffset;
    lastStartState_ = robotState;
    hasSolution_ = true;
    shiftRejected_ = false;
  } else if (guessWasShifted) {
    shiftRejected_ = true;
  }

  // A phase that has a plan keeps it unless the new one is reusable: a rejected re-solve must not stop a step in
  // progress, and one that is only usable for its first step must not move the footholds of that step. A rejected solve
  // (or an incomplete one) of a phase with no plan is published as a failure, without inputs.
  const bool keepNominal = phaseHasPlan && !reusable;
  Plan plan;
  if (!keepNominal) {
    plan.phase = static_cast<size_t>(std::max(gaitOffset, 0));
    plan.startState = robotState;
    if (accepted) {
      for (size_t k = 0; k < numKnots; ++k) {
        plan.inputs.push_back(extractRobotInput(outcome.solution.inputTrajectory_[k]));
      }
      plan.source = "nominal";
      if (!reusable) {
        // The later knots were solved from a state the first step does not lead to, and the controller runs knot 1 until
        // the next phase has its own plan. Each is kept only if, clamped to its bounds, it is a usable step from the
        // state the robot will really be in; otherwise it is a capture-point stop.
        vector_t state = outcome.successorState;
        for (size_t k = 1; k < numKnots; ++k) {
          vector_t input = saturateRobotInput(plan.inputs[k], params_);
          if (!saturatedStepUsable(state, input, params_)) {
            input = fallbackInput(params_, state, gaitOffset + static_cast<int>(k));
          }
          plan.inputs[k] = input;
          state = lipMapScalar(state, input, params_.omega(), params_.mass, params_.inertia);
        }
        plan.source = "first step";
      }
    } else {
      plan.failed = true;
      plan.source = "failed";
    }
    plan.trustworthy = outcome.planTrustworthy();
  }
  nominalPublished_ = accepted && !keepNominal;

  // Statistics first: a controller that reads the plan as soon as it is published then finds the statistics of its solve.
  SolveStatistics statistics;
  statistics.gaitOffset = gaitOffset;
  statistics.warmStart = warmStart;
  statistics.numIterations = solverPtr_->getIterationsLog().size();
  statistics.performance = solverPtr_->getPerformanceIndeces();
  statistics.source = keepNominal ? "rejected" : plan.source;
  statistics.pessiScale = acceptedScale;
  statistics.trustworthy = outcome.planTrustworthy();
  statistics.dynamicsResidual = outcome.dynamicsResidual;
  statistics.appliedViolation = outcome.constraintViolation;
  statistics.horizonViolation = outcome.horizonViolation;
  {
    std::lock_guard<std::mutex> lock(statisticsMutex_);
    lastSolveStatistics_ = statistics;
  }

  if (!keepNominal) {
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
