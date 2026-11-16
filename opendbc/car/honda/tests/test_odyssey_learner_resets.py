"""Kennesaw: brikowski's learner (OdysseyLearner) is off by default, resets on disengage and on a gear change, and keeps
only what he designed it to keep (coast calibration per gear)."""
import math
import unittest
from unittest.mock import patch

from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda import odyssey_learner as ol
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.honda.tests.test_odyssey_port import SimpleNamespace
from openpilot.common.params import Params


def controller(learner):
  CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
  CP_SP = CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], True, False, False)
  with patch.object(CarController, "_read_odyssey_learner", lambda self: learner):
    return CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, CP_SP)


def cs(v=20.0, a=0.0, gear=7, now=1_000_000_000):
  state = structs.CarState(vEgo=v, aEgo=a, canValid=True)
  fresh = now - 5_000_000  # received 5 ms ago: valid (stale inputs reset coast learning, by design)
  return SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0, odyssey_engine_torque_ts_nanos=fresh, odyssey_shift_activity=0,
                         odyssey_target_gear=gear, odyssey_user_brake=0., odyssey_computer_braking=False,
                         odyssey_computer_braking_ts_nanos=fresh, odyssey_target_gear_ts_nanos=fresh)


def control(long_active, accel=0.3):
  c = structs.CarControl(enabled=long_active, longActive=long_active, orientationNED=[0., 0., 0.])
  c.actuators.accel = accel
  c.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
  return c


def teach(learner):
  """Put state in every part of the learner, as after a few minutes of driving."""
  for frame in range(0, 120, 2):
    learner.gas_response.update(.3, .0, 0., 20., 7, True)
    learner.coast_response.update(frame, 7, 0., -.3, True, True)
  learner.gas_coast_release.update(0, -.1, .3, 20., -.03, 7, .2, True, True)
  learner.brake_response_error.x = .3
  assert learner.gas_response.integral != 0. and learner.gas_response.requests
  assert learner.coast_response.drag and learner.gas_coast_release.active


class TestOdysseyLearnerToggle(unittest.TestCase):
  def test_param_default_is_off(self):
    self.assertEqual(ol.ODYSSEY_LEARNER_PARAM, "OdysseyLearner")
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    CP_SP = CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], True, False, False)
    self.assertIsNone(CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, CP_SP).odyssey_learner)

  def test_param_on_selects_the_learner_for_the_odyssey_with_openpilot_long_only(self):
    Params().put_bool(ol.ODYSSEY_LEARNER_PARAM, True)
    try:
      for car, alpha_long, expected in ((CAR.HONDA_ODYSSEY_5G_MMR, True, True), (CAR.HONDA_ODYSSEY_5G_MMR, False, False),
                                        (CAR.HONDA_CIVIC_BOSCH, True, False)):
        with self.subTest(car=car, alpha_long=alpha_long):
          CP = CarInterface.get_params(car, gen_empty_fingerprint(), [], alpha_long, False, False)
          CP_SP = CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], alpha_long, False, False)
          c = CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, CP_SP)
          self.assertEqual(c.odyssey_learner is not None, expected)
    finally:
      Params().put_bool(ol.ODYSSEY_LEARNER_PARAM, False)

  def test_gas_map_floor_is_commas(self):
    # literal on purpose: the port replaced his CarControllerParams.BOSCH_GAS_LOOKUP_BP[0] with this constant
    self.assertEqual(ol.ODYSSEY_GAS_MAP_FLOOR, -0.2)


class TestOdysseyLearnerResets(unittest.TestCase):
  def test_disengage_resets_gas_learning_and_keeps_coast_calibration(self):
    c = controller(True)
    teach(c.odyssey_learner)
    c.update(control(False).as_reader(), structs.CarControlSP(), cs(), 1_000_000_000)
    lr = c.odyssey_learner
    self.assertEqual(lr.gas_response.integral, 0.)
    self.assertEqual(lr.gas_response.correction, 0.)
    self.assertFalse(lr.gas_response.requests)
    self.assertEqual(lr.brake_response_error.x, 0.)
    self.assertFalse(lr.gas_coast_release.active)
    self.assertTrue(lr.coast_response.drag, "coast calibration per gear is kept on purpose (his design)")
    self.assertFalse(c.odyssey_gas_selected or c.odyssey_brake_selected or c.odyssey_gas_bridge_active)

  def test_gear_change_restarts_the_gas_observer_and_the_coast_period(self):
    lr = ol.OdysseyLearnerState()
    teach(lr)
    lr.gas_response.update(.3, .0, 0., 20., 6, True)  # received target gear changed 7 -> 6
    self.assertEqual(len(lr.gas_response.requests), 1, "delayed requests from the old gear must not be compared")
    self.assertEqual(lr.gas_response.drivetrain, (6, 0))
    lr.coast_response.update(200, 6, 0., -.3, True, True)
    self.assertEqual(lr.coast_response.gear, 6)
    self.assertFalse(lr.coast_response.samples)

  def test_invalid_inputs_reset_everything(self):
    lr = ol.OdysseyLearnerState()
    teach(lr)
    lr.coast_response.update(200, 7, math.nan, -.3, True, True)
    self.assertFalse(lr.coast_response.drag)
    lr.gas_response.update(math.nan, .0, 0., 20., 7, True)
    self.assertFalse(lr.gas_response.requests)

  def test_off_never_touches_the_learner(self):
    c = controller(False)
    self.assertIsNone(c.odyssey_learner)
    for frame in range(0, 40, 2):
      c.update(control(True).as_reader(), structs.CarControlSP(), cs(), 1_000_000_000 + frame * 10_000_000)
    self.assertIsNone(c.odyssey_learner)


if __name__ == "__main__":
  unittest.main()
