#include <stdexcept>

#include <gtest/gtest.h>

#include "opti_pessi_interface/ObstacleSlots.h"
#include "opti_pessi_interface/OptiPessiReferenceManager.h"

using namespace opti_pessi;
using Vec2 = Eigen::Matrix<scalar_t, 2, 1>;

namespace {

constexpr scalar_t kGate = 1.0;
int nextId = 0;

std::vector<size_t> assign(std::vector<ObstacleSlot>& slots, const std::vector<Vec2>& centres) {
  return assignObstacleSlots(centres, slots, kGate, nextId);
}

OptiPessiReferenceManager makeReferenceManager(int numSlots) {
  OptiPessiModelParameters params;
  params.goal = vector_t::Zero(2);
  params.numSlots = numSlots;
  return OptiPessiReferenceManager(params);
}

ObservedObstacle human(scalar_t x, scalar_t y) { return ObservedObstacle{x, y, 0.2, 1.0}; }

}  // namespace

TEST(ObstacleSlots, NewObstaclesTakeTheFreeSlotsInOrder) {
  std::vector<ObstacleSlot> slots(4);
  EXPECT_EQ(assign(slots, {Vec2(1, 0), Vec2(2, 0)}), (std::vector<size_t>{0, 1}));
  EXPECT_TRUE(slots[0].taken);
  EXPECT_TRUE(slots[1].taken);
  EXPECT_FALSE(slots[2].taken);
  EXPECT_FALSE(slots[3].taken);
  EXPECT_DOUBLE_EQ(slots[1].x, 2.0);
}

TEST(ObstacleSlots, ObstaclesThatSwapRankKeepTheirSlots) {
  std::vector<ObstacleSlot> slots(4);
  assign(slots, {Vec2(2, 1), Vec2(2, -1.2)});
  // Perception sorts closest first: after the robot moved, the second one comes first in the message.
  EXPECT_EQ(assign(slots, {Vec2(2, -1.1), Vec2(2.1, 1)}), (std::vector<size_t>{1, 0}));
}

TEST(ObstacleSlots, AnObstacleThatMovesWithinTheGateKeepsItsSlot) {
  std::vector<ObstacleSlot> slots(4);
  assign(slots, {Vec2(1, 0), Vec2(5, 0), Vec2(9, 0)});
  EXPECT_EQ(assign(slots, {Vec2(9.6, 0.5)}), (std::vector<size_t>{2}));
  EXPECT_FALSE(slots[0].taken);
  EXPECT_FALSE(slots[1].taken);
  EXPECT_DOUBLE_EQ(slots[2].x, 9.6);
}

TEST(ObstacleSlots, ANewObstacleTakesASlotThatWasFreeBeforeOneJustLeft) {
  std::vector<ObstacleSlot> slots(3);
  assign(slots, {Vec2(1, 0), Vec2(5, 0)});
  // The obstacle of slot 0 is gone and one appears far from both: slot 2 was free, slot 0 has only just been left.
  EXPECT_EQ(assign(slots, {Vec2(5, 0), Vec2(-3, 0)}), (std::vector<size_t>{1, 2}));
}

TEST(ObstacleSlots, ANewObstacleTakesTheSlotOfOneGoneWhenNoneIsFree) {
  std::vector<ObstacleSlot> slots(2);
  assign(slots, {Vec2(1, 0), Vec2(5, 0)});
  EXPECT_EQ(assign(slots, {Vec2(5, 0), Vec2(-3, 0)}), (std::vector<size_t>{1, 0}));
}

TEST(ObstacleSlots, BeyondTheGateAnObstacleCountsAsNew) {
  std::vector<ObstacleSlot> slots(2);
  assign(slots, {Vec2(1, 0)});
  EXPECT_EQ(assign(slots, {Vec2(2.5, 0)}), (std::vector<size_t>{1}));
}

TEST(ObstacleSlots, TheNearerOfTwoObstaclesTakesTheSlot) {
  std::vector<ObstacleSlot> slots(2);
  assign(slots, {Vec2(0, 0)});
  EXPECT_EQ(assign(slots, {Vec2(0.6, 0), Vec2(0.2, 0)}), (std::vector<size_t>{1, 0}));
}

TEST(ObstacleSlots, NoObstacleFreesEverySlot) {
  std::vector<ObstacleSlot> slots(2);
  assign(slots, {Vec2(1, 0), Vec2(2, 0)});
  EXPECT_TRUE(assign(slots, {}).empty());
  EXPECT_FALSE(slots[0].taken);
  EXPECT_FALSE(slots[1].taken);
}

TEST(ObstacleSlots, AnObstacleKeepsItsIdAndANewOneGetsTheNext) {
  std::vector<ObstacleSlot> slots(2);
  int ids = 7;
  assignObstacleSlots({Vec2(1, 0)}, slots, kGate, ids);
  EXPECT_EQ(slots[0].id, 7);
  // The obstacle of slot 0 goes and a new one takes its slot: same slot, new id.
  assignObstacleSlots({Vec2(1.3, 0), Vec2(-4, 0)}, slots, kGate, ids);
  EXPECT_EQ(slots[0].id, 7);
  EXPECT_EQ(slots[1].id, 8);
  assignObstacleSlots({Vec2(-4, 0.2)}, slots, kGate, ids);
  EXPECT_EQ(slots[1].id, 8);
  assignObstacleSlots({Vec2(-4, 0.2), Vec2(1, 3)}, slots, kGate, ids);
  EXPECT_EQ(slots[0].id, 9);
  EXPECT_EQ(ids, 10);
}

TEST(ObstacleSlots, MoreObstaclesThanSlotsThrows) {
  std::vector<ObstacleSlot> slots(1);
  EXPECT_THROW(assign(slots, {Vec2(1, 0), Vec2(2, 0)}), std::invalid_argument);
}

TEST(ReferenceManagerObstacles, EverySlotIsFreeBeforeTheFirstObservation) {
  const auto referenceManager = makeReferenceManager(2);
  const vector_t p = referenceManager.getObstacleParameters(2);
  EXPECT_DOUBLE_EQ(p(ObstacleP::ACTIVE), 0.0);
  EXPECT_DOUBLE_EQ(p(ObstacleP::DIM + ObstacleP::ACTIVE), 0.0);
}

TEST(ReferenceManagerObstacles, FreeSlotsAreInactiveAndParked) {
  auto referenceManager = makeReferenceManager(3);
  const vector_t com = (vector_t(2) << 1.0, 2.0).finished();
  EXPECT_EQ(referenceManager.setObservedObstacles({human(4, 0)}, com), 0u);
  const vector_t p = referenceManager.getObstacleParameters(3);
  EXPECT_DOUBLE_EQ(p(ObstacleP::X), 4.0);
  EXPECT_DOUBLE_EQ(p(ObstacleP::RADIUS), 0.2);
  EXPECT_DOUBLE_EQ(p(ObstacleP::MAX_SPEED), 1.0);
  EXPECT_DOUBLE_EQ(p(ObstacleP::ACTIVE), 1.0);
  for (int j = 1; j < 3; ++j) {
    EXPECT_DOUBLE_EQ(p(ObstacleP::DIM * j + ObstacleP::ACTIVE), 0.0);
    EXPECT_DOUBLE_EQ(p(ObstacleP::DIM * j + ObstacleP::X), 1.0 + OptiPessiReferenceManager::kParkingDistance);
    EXPECT_DOUBLE_EQ(p(ObstacleP::DIM * j + ObstacleP::Y), 2.0);
    EXPECT_DOUBLE_EQ(p(ObstacleP::DIM * j + ObstacleP::RADIUS), 0.0);
    EXPECT_DOUBLE_EQ(p(ObstacleP::DIM * j + ObstacleP::MAX_SPEED), 0.0);
  }
}

TEST(ReferenceManagerObstacles, TheClosestKeepOutDisksTakeTheSlots) {
  auto referenceManager = makeReferenceManager(2);
  const vector_t com = vector_t::Zero(2);
  // A car 3 m away comes closer with its 2.4 m disk than a person 1.5 m away.
  const ObservedObstacle car{3.0, 0.0, 2.4, 2.0};
  EXPECT_EQ(referenceManager.setObservedObstacles({human(0, 5), human(0, 1.5), car}, com), 1u);
  // New obstacles fill the free slots closest first.
  const matrix_t& obstacles = referenceManager.getObstacles();
  EXPECT_DOUBLE_EQ(obstacles(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(obstacles(1, 1), 1.5);
}

TEST(ReferenceManagerObstacles, AnObstacleKeepsItsSlotAcrossCalls) {
  auto referenceManager = makeReferenceManager(4);
  const vector_t com = vector_t::Zero(2);
  referenceManager.setObservedObstacles({human(2, 1), human(2, -1.2)}, com);
  referenceManager.setObservedObstacles({human(2, -1.1), human(2.1, 1)}, com);
  const matrix_t& obstacles = referenceManager.getObstacles();
  EXPECT_DOUBLE_EQ(obstacles(0, 0), 2.1);
  EXPECT_DOUBLE_EQ(obstacles(1, 1), -1.1);
}

TEST(ReferenceManagerObstacles, ASmallerOcpGetsTheOccupiedSlotsFirst) {
  auto referenceManager = makeReferenceManager(4);
  const vector_t com = vector_t::Zero(2);
  referenceManager.setObservedObstacles({human(2, 1), human(5, 0), human(-3, 0)}, com);
  referenceManager.setObservedObstacles({human(5.1, 0), human(-3, 0.1)}, com);  // the one in slot 0 is gone
  EXPECT_EQ(referenceManager.numActiveObstacles(), 2);

  const vector_t p = referenceManager.getObstacleParameters(2);
  EXPECT_DOUBLE_EQ(p(ObstacleP::X), 5.1);
  EXPECT_DOUBLE_EQ(p(ObstacleP::DIM + ObstacleP::X), -3.0);
  EXPECT_DOUBLE_EQ(p(ObstacleP::ACTIVE), 1.0);
  EXPECT_DOUBLE_EQ(p(ObstacleP::DIM + ObstacleP::ACTIVE), 1.0);

  const std::vector<int> ids2 = referenceManager.getObstacleIds(2);
  const std::vector<int> ids4 = referenceManager.getObstacleIds(4);
  EXPECT_EQ(ids2.size(), 2u);
  EXPECT_EQ(ids4[0], ids2[0]);
  EXPECT_EQ(ids4[1], ids2[1]);
  EXPECT_EQ(ids4[2], -1);
  EXPECT_EQ(ids4[3], -1);
  EXPECT_NE(ids2[0], ids2[1]);
}

TEST(ReferenceManagerObstacles, AnOcpLargerThanTheObstaclesGetsFreeSlots) {
  auto referenceManager = makeReferenceManager(2);
  referenceManager.setObservedObstacles({}, vector_t::Zero(2));
  EXPECT_EQ(referenceManager.numActiveObstacles(), 0);
  const vector_t p = referenceManager.getObstacleParameters(1);
  EXPECT_DOUBLE_EQ(p(ObstacleP::ACTIVE), 0.0);
  EXPECT_EQ(referenceManager.getObstacleIds(1), std::vector<int>{-1});
}
