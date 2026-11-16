"""OdysseyRdmOff (Jasper, MVL odytest 0a3644502 by content): off sends LKAS_HUD exactly as before; on, under openpilot
long, the single LKAS_HUD carries RDM_OFF and LANE_ASSIST_BEEP_OFF."""
import math
import unittest

from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda.carcontroller import CarController, ODYSSEY_RDM_OFF_PARAM
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.honda.tests.test_odyssey_port import SimpleNamespace  # fake CarState backed by a real one
from openpilot.common.params import Params

LKAS_HUD = 0x33D


def _sp(CP):
  return CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], CP.openpilotLongitudinalControl, False, False)


def hud_bits(dat):
  return (dat[3] >> 3) & 1, (dat[3] >> 6) & 1  # RDM_OFF (bit 27), LANE_ASSIST_BEEP_OFF (bit 30)


def van_fingerprint():
  fp = gen_empty_fingerprint()
  fp[0][0x33DA] = 8  # the van's extended HUD (BOSCH_EXT_HUD, flags 519 on its drives); an empty fingerprint misses it
  return fp


def run(alpha_long, rdm_off, car=CAR.HONDA_ODYSSEY_5G_MMR, frames=40):
  Params().put_bool(ODYSSEY_RDM_OFF_PARAM, rdm_off)
  try:
    CP = CarInterface.get_params(car, van_fingerprint(), [], alpha_long, False, False)
    assert CP.flags & 1, "BOSCH_EXT_HUD"
    controller = CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, _sp(CP))
  finally:
    Params().put_bool(ODYSSEY_RDM_OFF_PARAM, False)
  control = structs.CarControl()
  control.enabled = control.latActive = True
  state = structs.CarState()
  state.vEgo = 15.0
  state.cruiseState.available = True
  CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={'LKAS_PROBLEM': 0},
                       stock_brake={}, odyssey_car_gas=math.nan, odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0,
                       odyssey_target_gear=0, odyssey_user_brake=0., odyssey_computer_braking=False,
                       odyssey_computer_braking_ts_nanos=0)
  huds = []
  for frame in range(frames):
    _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
    huds += [bytes(dat) for addr, dat, _ in sends if addr == LKAS_HUD]
  return controller, huds


class TestOdysseyRdmOff(unittest.TestCase):
  def test_off_is_todays_hud(self):
    controller, huds = run(alpha_long=True, rdm_off=False)
    self.assertFalse(controller.odyssey_rdm_off)
    self.assertTrue(huds)
    self.assertTrue(all(hud_bits(d) == (0, 0) for d in huds))  # as logged on 00000086--791f5e1d12 (Ideal)

  def test_on_under_openpilot_long_sets_rdm_off(self):
    controller, huds = run(alpha_long=True, rdm_off=True)
    self.assertTrue(controller.odyssey_rdm_off)
    self.assertTrue(huds)
    self.assertTrue(all(hud_bits(d) == (1, 1) for d in huds))

  def test_on_without_openpilot_long_does_nothing(self):
    controller, _ = run(alpha_long=False, rdm_off=True)
    self.assertFalse(controller.odyssey_rdm_off)

  def test_other_hondas_ignore_the_param(self):
    controller, _ = run(alpha_long=True, rdm_off=True, car=CAR.HONDA_CIVIC_BOSCH)
    self.assertFalse(controller.odyssey_rdm_off)

  def test_off_hud_bytes_identical_to_on_apart_from_the_two_bits(self):
    _, off = run(alpha_long=True, rdm_off=False)
    _, on = run(alpha_long=True, rdm_off=True)
    self.assertEqual(len(off), len(on))
    for a, b in zip(off, on, strict=True):
      self.assertEqual(a[:3], b[:3])
      self.assertEqual(a[3] & ~0x48 & 0xFF, b[3] & ~0x48 & 0xFF)


if __name__ == "__main__":
  unittest.main()
