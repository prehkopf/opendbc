from openpilot.common.prefix import OpenpilotPrefix

# The Honda controller reads openpilot Params when it is built (the Odyssey toggles such as OdysseyRdmOff and
# OdysseyLearner, the learned gas and speed factors) and its param writer thread writes them back. Params live in one
# directory per OPENPILOT_PREFIX, so without a prefix every pytest-xdist worker shares one directory: a test that turns
# a toggle on while it builds its controller races another worker's controller, which then starts with the wrong
# setting. Give each test process its own params directory. pytest_configure runs in every xdist worker, before
# collection.
_prefix = None


def pytest_configure(config):
  global _prefix
  _prefix = OpenpilotPrefix()
  _prefix.__enter__()


def pytest_unconfigure(config):
  global _prefix
  if _prefix is not None:
    _prefix.__exit__(None, None, None)
    _prefix = None
