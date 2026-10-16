// The random-walk obstacle of the plant: RandomObstacle of opti_pessi_acados' obstacle_manager.py.
#include <gtest/gtest.h>

#include <cmath>

#include "opti_pessi_interface/simulation/ObstaclePlant.h"

using namespace opti_pessi;

namespace {

constexpr scalar_t kDt = 0.1;

// Two random obstacles in the corridor x in [1, 9], y in [-1, 1]: one heading along +x, one along +y.
OptiPessiModelParameters randomScenario(unsigned int seed) {
  OptiPessiModelParameters p;
  p.obstaclePositions = (matrix_t(2, 2) << 2.0, 0.0, 5.0, 0.5).finished();
  p.obstacleMovement = {ObstacleMovement::Random, ObstacleMovement::Random};
  p.obstacleSpeed = (vector_t(2) << 0.4, 0.3).finished();
  p.obstacleDir = (matrix_t(2, 2) << 1.0, 0.0, 0.0, 1.0).finished();
  p.randomBounds = (vector_t(4) << 1.0, 9.0, -1.0, 1.0).finished();
  p.randomSeed = seed;
  return p;
}

// Positions after each of `steps` steps, one column per step, obstacle j in rows 2j, 2j+1.
matrix_t walk(const OptiPessiModelParameters& p, int steps) {
  ObstaclePlant plant = makeObstaclePlant(p);
  matrix_t path(2 * p.numPlantObstacles(), steps);
  for (int n = 0; n < steps; ++n) {
    stepObstacles(plant, p, vector_t::Zero(2), kDt, n);
    for (int j = 0; j < p.numPlantObstacles(); ++j) {
      path.col(n).segment(2 * j, 2) = plant.positions.row(j).transpose();
    }
  }
  return path;
}

}  // namespace

TEST(RandomObstacle, FirstStepIsItsSpeedAlongItsDirectionGivenOrTakeTheNoise) {
  const auto p = randomScenario(1234);
  ObstaclePlant plant = makeObstaclePlant(p);
  stepObstacles(plant, p, vector_t::Zero(2), kDt, 0);
  for (int j = 0; j < p.numPlantObstacles(); ++j) {
    const vector_t step = (plant.positions.row(j) - p.obstaclePositions.row(j)).transpose();
    EXPECT_NEAR(step.norm(), p.obstacleSpeed(j) * kDt, 1e-12);
    // The heading noise is at most 1 rad/s.
    const scalar_t turn = std::acos(std::min(1.0, step.normalized().dot(p.obstacleDir.row(j).transpose())));
    EXPECT_LE(turn, 1.0 * kDt + 1e-9);
  }
}

TEST(RandomObstacle, StaysInsideItsBoundsAndNeverOutrunsItsSpeed) {
  const auto p = randomScenario(1234);
  const matrix_t path = walk(p, 3000);
  const vector_t& b = p.randomBounds;
  for (int j = 0; j < p.numPlantObstacles(); ++j) {
    vector_t previous = p.obstaclePositions.row(j).transpose();
    for (int n = 0; n < path.cols(); ++n) {
      const vector_t now = path.col(n).segment(2 * j, 2);
      EXPECT_GE(now(0), b(0));
      EXPECT_LE(now(0), b(1));
      EXPECT_GE(now(1), b(2));
      EXPECT_LE(now(1), b(3));
      EXPECT_LE((now - previous).norm(), p.obstacleSpeed(j) * kDt + 1e-12);
      previous = now;
    }
  }
}

TEST(RandomObstacle, SameSeedSamePathOtherSeedOtherPath) {
  const matrix_t a = walk(randomScenario(7), 200);
  EXPECT_EQ((a - walk(randomScenario(7), 200)).cwiseAbs().maxCoeff(), 0.0);
  EXPECT_GT((a - walk(randomScenario(8), 200)).cwiseAbs().maxCoeff(), 1e-3);
}

TEST(RandomObstacle, BouncesOffAWall) {
  auto p = randomScenario(1234);
  p.obstaclePositions(0, 0) = 8.98;  // heading along +x, 2 cm from x = 9
  ObstaclePlant plant = makeObstaclePlant(p);
  stepObstacles(plant, p, vector_t::Zero(2), kDt, 0);
  EXPECT_EQ(plant.positions(0, 0), 9.0);  // clipped onto the wall
  stepObstacles(plant, p, vector_t::Zero(2), kDt, 1);
  EXPECT_LT(plant.positions(0, 0), 9.0 - 0.3 * kDt);  // and back, nearly at full speed
}
