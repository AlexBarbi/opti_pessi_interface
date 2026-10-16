#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <ocs2_ipm/IpmSolver.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>

#include "opti_pessi_interface/OptiPessiInterface.h"
#include "opti_pessi_interface/OptiPessiModelParameters.h"
#include "opti_pessi_interface/OptiPessiReferenceManager.h"

namespace opti_pessi {

/**
 * IpmMpc for the Opti-Pessi OCP, run through MPC_MRT_Interface.
 *
 * Every solve starts at knot 0 (see definitions.h) and is warm-started from the previous solution: as it is while the
 * phase is the same, shifted one knot on the next phase (shiftPrimalSolution: its knot-0 footholds are the ones the
 * current stance feet already stand on, and reusing it unshifted would put both feet of each side in the same place).
 * If the solver fails on that guess it is run once more from a cold start. Whatever the solver returns is published
 * (getLatestPlan()) and becomes the next warm start: the solution is not checked against the dynamics or the constraints,
 * and nothing replaces it.
 *
 * "The solver fails" means it threw, or returned no complete, finite trajectory. When the cold start fails too there is
 * nothing to publish and the previous plan stays. The first solve after reset() has no previous solution and starts cold.
 * The MRT buffer only feeds visualization.
 *
 * Every solve uses the OCP with one slot per obstacle in the reference manager (OptiPessiInterface::ocpSlots()), each
 * size with its own solver: a slot costs as much free as taken. When the obstacles in the slots change (one more, one
 * less, one replaced), the warm start keeps every remaining obstacle's planes, moved to its new slot, and seeds those of
 * a new one (remapObstacleSlots()); the solver then starts its slacks and duals afresh.
 */
class OptiPessiMpc final : public ocs2::MPC_BASE {
 public:
  /** Solvers for every OCP of `interface`, which must outlive this object. */
  explicit OptiPessiMpc(const OptiPessiInterface& interface);

  ~OptiPessiMpc() override = default;

  /** The solver of the latest solve. */
  ocs2::IpmSolver* getSolverPtr() override { return solvers_[activeSolver_].get(); }
  const ocs2::IpmSolver* getSolverPtr() const override { return solvers_[activeSolver_].get(); }

  /** Binds every solver to `referenceManagerPtr` (they all read the one reference manager of the interface). */
  void setReferenceManager(std::shared_ptr<ocs2::ReferenceManagerInterface> referenceManagerPtr);

  void reset() override;

  /** Solves, then returns whether a plan was published (MPC_MRT_Interface then copies it for drawing). */
  bool run(scalar_t currentTime, const vector_t& currentState) override;

  /** A plan for one contact phase, in robot coordinates. */
  struct Plan {
    size_t sequence = 0;           // increments with every published plan; 0 before the first
    size_t phase = 0;              // contact phase (gait offset) it was solved for
    vector_t startState;           // measured 10-dof LIP state it was solved from
    std::vector<vector_t> inputs;  // robot inputs of knots 0..N-1, as the solver returned them
    const char* source = "none";   // SolveStatistics::warmStart of the solve it comes from
  };

  /** Sequence number of the latest published plan, cheap to poll from any thread. */
  size_t getPlanSequence() const { return planSequence_; }

  /** Latest published plan, safe to read from any thread. */
  Plan getLatestPlan() const;

  /** Outcome of the latest solve, for diagnostics. */
  struct SolveStatistics {
    int gaitOffset = 0;
    const char* warmStart = "none";      // "cold", "shifted" (new phase), "same phase", or "cold retry" after a failed warm start
    int obstacleSlots = 0;               // slots of the OCP solved
    bool obstaclesChanged = false;       // other obstacles in the slots than in the warm start: its planes were remapped
    bool solved = false;                 // the solver returned a trajectory and it was published
    size_t numIterations = 0;            // of the last run
    ocs2::PerformanceIndex performance;  // the solver's own figures for the last run
  };

  /** Written on the MPC thread after every solve, safe to read from any thread. */
  SolveStatistics getLastSolveStatistics() const;

  /** Totals over every solve since construction: reset() does not clear them. */
  struct SolveTotals {
    size_t solves = 0;
    size_t samePhase = 0;         // warm-started from the solution of the same phase
    size_t shifted = 0;           // warm-started from the previous phase's solution, shifted
    size_t cold = 0;              // started cold: no previous solution
    size_t coldRetries = 0;       // cold starts after the solver failed on the warm start
    size_t obstacleChanges = 0;   // warm starts remapped to other obstacles in the slots
    size_t unsolved = 0;          // the solver failed on every start: nothing published
    size_t iterations = 0;        // of the last run of each solve
    size_t atIterationCap = 0;    // solves whose last run stopped at ipmIteration
    scalar_t solveTimeSum = 0.0;  // [ms], wall clock, retries included
    scalar_t solveTimeMax = 0.0;  // [ms]
    std::string solverProfile;    // ocs2's own profile of the solver, over its runs since its last reset (a cold start)
  };

  /** Safe to read from any thread. */
  SolveTotals getSolveTotals() const;

 protected:
  void calculateController(scalar_t initTime, const vector_t& initState, scalar_t finalTime) override;

 private:
  /**
   * One run of the active solver from `guess`, or cold without one; `freshDuals` restarts its slacks and duals.
   * False if the solver failed (see the class comment).
   */
  bool solve(const vector_t& robotState, const ocs2::PrimalSolution* guess, bool freshDuals, ocs2::PrimalSolution& solution);

  const OptiPessiInterface& interface_;
  std::vector<std::unique_ptr<ocs2::IpmSolver>> solvers_;  // index numSlots - 1
  size_t activeSolver_ = 0;            // solver of the latest solve
  std::shared_ptr<OptiPessiReferenceManager> referenceManagerPtr_;
  OptiPessiModelParameters params_;
  ocs2::PrimalSolution lastSolution_;  // last solution published, the next warm start
  std::vector<int> lastObstacleIds_;   // obstacle in each slot of lastSolution_'s OCP
  bool hasSolution_ = false;
  bool published_ = false;             // the latest solve was published, see run()
  int lastGaitOffset_ = 0;             // phase lastSolution_ was solved for
  vector_t lastStartState_;            // measured LIP state lastSolution_ was solved from
  bool solverFailed_ = false;          // the solver's last run failed: its slacks and duals are not those of lastSolution_
  size_t iterationCap_ = 0;            // ipm.ipmIteration

  mutable std::mutex planMutex_;
  Plan latestPlan_;
  std::atomic<size_t> planSequence_{0};

  mutable std::mutex statisticsMutex_;
  SolveStatistics lastSolveStatistics_;
  SolveTotals totals_;
};

}  // namespace opti_pessi
