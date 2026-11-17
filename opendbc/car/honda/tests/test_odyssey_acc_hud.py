import unittest

from opendbc.can import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.honda import hondacan
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC

# Stock ACC_HUD bits the Nidec path copies from the car; all clear, so a set bit below comes from create_acc_hud.
_STOCK_ACC_HUD = {'FCM_OFF': 0, 'FCM_OFF_2': 0, 'FCM_PROBLEM': 0, 'ICONS': 0}


def _acc_hud_values(car):
  """ACC_HUD as openpilot sends it under openpilot longitudinal, decoded with the car's own DBC."""
  CP = CarInterface.get_params(car, gen_empty_fingerprint(), [], True, False, False)
  dbc = DBC[car][Bus.pt]
  bus = hondacan.CanBus(CP).pt
  addr, dat, _ = hondacan.create_acc_hud(CANPacker(dbc), bus, CP, True, 0., 0., structs.CarControl.HUDControl(), 0, False,
                                         _STOCK_ACC_HUD, 0, True)
  parser = CANParser(dbc, [("ACC_HUD", 0)], bus)
  parser.update([(0, [(addr, dat, bus)])])
  return parser.vl["ACC_HUD"]


class TestOdysseyAccHud(unittest.TestCase):
  def test_bosch_openpilot_long_shows_cmbs_fcw_off(self):
    # openpilot longitudinal silences the Bosch radar (interface.py init), so CMBS/FCW are off and the dash says so (MVL a9c8f53)
    values = _acc_hud_values(CAR.HONDA_ODYSSEY_5G_MMR)
    self.assertEqual((values['FCM_OFF'], values['FCM_OFF_2']), (1, 1))
    self.assertEqual(values['ACC_ON'], 1)


if __name__ == "__main__":
  unittest.main()
