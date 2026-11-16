"""RehkopfPilot (Kennesaw): brikowski's Odyssey learner of 2026-10-21 to 11-01, ported by content.

Param OdysseyLearner, default off. Off, the Honda controller runs exactly as in Jasper and nothing here is used.

Source: brikowski/opendbc ody-op-test c91503afe (2026-11-01, the opendbc pin of his 11-02 ody-op stable), the
Odyssey helpers and the Odyssey branch of CarController.update, taken by file content. Credit: brikowski.
What his 21 commits on top of 9d7186906 add (themes):
  1. gas calibration through coast: a learned gas reduction survives coast and is rebased to the gas headroom
     when gas resumes (554d5ee23); added gas is withdrawn on sustained excess response (4deda8a84, 0c2747424)
  2. warmup: issued corrections are withdrawn in proportion to the reverse error instead of dropped (49d8c1d8d,
     bd99f2b80)
  3. coast learning: one sample buffer per uninterrupted coast (a741ce343, 60829157f), calibration kept through
     idle stop and between observations (852c47a9b, 19d433ef1), reset when its inputs go stale
  4. exhausted gas released into coast on flat roads too, not only downhill (dc8c6f148, 963d3a843, 0573073eb)
  5. brake handoffs: excess braking released within the coast forecast margin (ba58977e0, c1b897f57), brake
     response error filtered so one wheel-acceleration dip can't override a settled forecast
  6. uphill: grade load added before calibration, one ceiling through zero demand (27628fdce, a8d758d88's
     feedforward), gas cap weight paired with the delayed request it belongs to (c30238e6f)
  7. correctness: encoded gas counts checked as the packer rounds them (7a46c8cbb), gas feedback only with the
     brakes released (b67482512), passive-response speed boundary shared at 5 m/s (c91503afe)
Kept ours (rehkopfpilot.md rules): safety flag 256 and panda safety untouched, ODYSSEY_GAS_MAP_FLOOR for comma's
gas map floor, the van-load factor on the gas map input, 0.5 s delay, 3840 steering, MVL's base.
Not taken: CarState.ecoMode / ecoModeValid (car.capnp fields; his ECON-change learner reset waits for an SP-struct
ECON signal), his fractional steering slew state (24057b269, lateral; its own toggle later), and moving the gas map
to instance level (a8d758d88, test isolation only).
"""
import math
from collections import deque

import numpy as np

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, DT_CTRL, structs
from opendbc.car.common.filter_simple import FirstOrderFilter

LongCtrlState = structs.CarControl.Actuators.LongControlState

ODYSSEY_LEARNER_PARAM = "OdysseyLearner"
# comma's Bosch gas map starts at -0.2 m/s^2; brikowski's helpers read it from CarControllerParams, which is MVL's
# 0.0 here until the Odyssey get_params overwrites it (tools/ody README, "Tests ported from brikowski").
ODYSSEY_GAS_MAP_FLOOR = -0.2

ODYSSEY_LOW_SPEED_DOMAIN_VEGO = 5.0
# Include the float32 representation of a nominal -0.10 m/s2 carControl request.
ODYSSEY_GAS_BRIDGE_ENTRY = -0.101
ODYSSEY_GAS_BRIDGE_COMMAND = -60.0
ODYSSEY_GRADE_FILTER_TAU = 0.5
ODYSSEY_GRADE_RAMP_ACCEL = 0.30
ODYSSEY_GRADE_GAIN = 0.6
ODYSSEY_UPHILL_GAS_ACCEL_MAX = 1.0
ODYSSEY_RESPONSE_ERROR_TAU = 0.3
ODYSSEY_RESPONSE_DELAY_FRAMES = 25  # 0.5 s Bosch longitudinal delay at 50 Hz
ODYSSEY_RESPONSE_MAX_COUNTS = 100.0
ODYSSEY_RESPONSE_COUNTS_PER_ACCEL = 200.0
ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL = 500.0
ODYSSEY_RESPONSE_SLEW_COUNTS = 10.0
ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS = 200.0
ODYSSEY_REQUEST_LOOKBACK_NS = 200_000_000
ODYSSEY_COAST_RELEASE_GRAVITY = 0.25
ODYSSEY_COAST_RELEASE_ERROR = 0.20
# Keep mild negative road-speed requests out of friction braking. Brake selection remains based on
# the raw request; gas continuity and response-qualified coast release never add friction braking.
ODYSSEY_ROAD_BRAKE_ENTRY = -0.30


def odyssey_command_domains(accel, speed, previous_brake=False, previous_gas=False, bridge_active=False,
                            gas_accel=None, release_brake=False, release_gas=False, passive_accel=None):
  """Keep low-speed stop authority and separate road-speed coast from friction braking."""
  gas_accel = accel if gas_accel is None else gas_accel
  gas_selected = accel > 0.0
  if speed >= ODYSSEY_LOW_SPEED_DOMAIN_VEGO:
    gas_selected |= (not previous_brake and not previous_gas and ODYSSEY_GAS_BRIDGE_ENTRY <= accel < 0.0 and
                     gas_accel > ODYSSEY_GAS_MAP_FLOOR)
    gas_selected |= previous_gas and min(accel, gas_accel) > ODYSSEY_GAS_MAP_FLOOR
    if bridge_active and accel < ODYSSEY_GAS_BRIDGE_ENTRY:
      gas_selected = False
    brake_selected = not release_brake and (accel < ODYSSEY_ROAD_BRAKE_ENTRY or (previous_brake and accel < 0.0))
    # Holding net acceleration can still need brakes when coasting would accelerate faster.
    if previous_brake and not release_brake and passive_accel is not None and accel < passive_accel:
      gas_selected, brake_selected = False, True
  else:
    brake_selected = accel <= 0.0
  if release_gas:
    gas_selected = False
  # Gas and brake commands remain mutually exclusive.
  return gas_selected, brake_selected and not gas_selected


def odyssey_gas_command(accel, mapped_gas, gas_selected, previous_gas, bridge_active, grade_weight=0.0):
  """Pre-activate gas at a grade-scaled command while keeping the level-road bridge."""
  bridge_active = gas_selected and accel < 0.0 and (bridge_active or not previous_gas)
  gas = ODYSSEY_GAS_BRIDGE_COMMAND + grade_weight * (mapped_gas - ODYSSEY_GAS_BRIDGE_COMMAND) if bridge_active else mapped_gas
  return (gas if gas_selected else 0.0), bridge_active


def odyssey_grade_weight(pitch):
  grade = max(0.0, math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY)
  load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
  return load * load


def odyssey_grade_gas_accel(accel, pitch):
  """Add the grade load before gas calibration and its feedforward ceiling."""
  grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY
  if grade < 0.0:
    # Gravity still supplies acceleration when the requested net acceleration is zero.
    return max(ODYSSEY_GAS_MAP_FLOOR, accel + grade * ODYSSEY_GRADE_GAIN)
  zero_grade = min(grade * ODYSSEY_GRADE_GAIN * odyssey_grade_weight(pitch), ODYSSEY_UPHILL_GAS_ACCEL_MAX)
  if zero_grade > 0.0 and ODYSSEY_ROAD_BRAKE_ENTRY < accel <= 0.0:
    gas_split = ODYSSEY_GAS_MAP_FLOOR
    if accel <= gas_split:
      x = (accel - ODYSSEY_ROAD_BRAKE_ENTRY) / (gas_split - ODYSSEY_ROAD_BRAKE_ENTRY)
      zero_grade *= x * x * (3.0 - 2.0 * x)
    # Preserve request sensitivity through zero, even on a steep climb.
    return accel + zero_grade
  if accel <= 0.0:
    return accel
  x = min(accel / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
  grade_weight = x * x * (3.0 - 2.0 * x)
  grade_accel = zero_grade + (grade * ODYSSEY_GRADE_GAIN - zero_grade) * grade_weight
  return accel + grade_accel


def odyssey_creep_brake_accel(accel, speed):
  """Compensate weak creep-speed braking without retaining a released request."""
  if not math.isfinite(speed) or speed < 0.0 or accel >= 0.0:
    return accel
  correction = float(np.interp(accel, [-0.8, -0.3, -0.1, 0.0], [0.0, -0.25, -0.25, 0.0]))
  weight = float(np.interp(speed, [1.0, 2.0], [1.0, 0.0]))
  return accel + weight * correction


def odyssey_low_speed_gas_command(gas, accel, speed):
  """Bound positive-gas feedforward where Odyssey acceleration exceeds the net request."""
  if accel <= 0.0 or gas <= 0.0:
    return gas

  def smoothstep(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

  speed_weight = smoothstep((speed - 8.0) / 4.0) * smoothstep((24.0 - speed) / 4.0)
  request_weight = smoothstep((accel - 0.4) / 0.4) * smoothstep((2.0 - accel) / 0.4)
  return max(0.0, gas - ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS * speed_weight * request_weight)


def odyssey_load_gas_accel(gas_accel, load_factor):
  """RehkopfPilot van load: scale the positive (force-producing) gas map input by mass (as carcontroller.py)."""
  return gas_accel * load_factor if gas_accel > 0.0 else gas_accel


def odyssey_gas_feedforward(accel, gas_accel, speed, params, load_factor=1.0):
  """Apply gas calibration and its uphill ceiling together, before response feedback."""
  def calibrated(map_accel, request):
    # RehkopfPilot: the van-load factor scales the gas map input, as in today's controller
    gas = float(np.interp(odyssey_load_gas_accel(map_accel, load_factor), params.BOSCH_GAS_LOOKUP_BP, params.BOSCH_GAS_LOOKUP_V))
    return odyssey_low_speed_gas_command(gas, request, speed)

  gas = calibrated(gas_accel, accel)
  if gas_accel <= accel:
    return gas, 0.0
  ceiling_request = max(accel, ODYSSEY_UPHILL_GAS_ACCEL_MAX)
  capped = min(gas, calibrated(ceiling_request, ceiling_request))
  cap_weight = 0.0
  if accel > 0.0 and capped < gas:
    base = calibrated(accel, accel)
    if gas > base:
      limited_fraction = float(np.clip((gas - capped) / (gas - base), 0.0, 1.0))
      load = min((gas_accel - accel) / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
      cap_weight = limited_fraction * load * load * (3.0 - 2.0 * load)
  return capped, cap_weight


class OdysseyGasResponse:
  """Compare delayed gas requests and response within the same received drivetrain state."""
  def __init__(self):
    self.correction = 0.0
    self.integral = 0.0
    self.requests = deque(maxlen=ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
    self.pitch_peak = None
    self.drivetrain = None
    self.paused = False
    self.error = FirstOrderFilter(0.0, ODYSSEY_RESPONSE_ERROR_TAU, DT_CTRL * 2, initialized=False)

  def reset(self):
    self.__init__()

  def clear_observer(self):
    self.integral = 0.0
    self.requests.clear()
    self.pitch_peak = None
    self.paused = False
    self.error = FirstOrderFilter(0.0, ODYSSEY_RESPONSE_ERROR_TAU, DT_CTRL * 2, initialized=False)

  def update(self, request, aego, pitch, speed, gear, eligible, cap_weight=0.0, shift_activity=0,
             limits=(-ODYSSEY_RESPONSE_MAX_COUNTS, ODYSSEY_RESPONSE_MAX_COUNTS), coast=False):
    if (not (eligible or coast) or not all(math.isfinite(v) for v in (request, aego, pitch, speed, gear, cap_weight, *limits)) or
        not 1 <= gear <= 10):
      self.reset()
      return 0.0

    drivetrain = (gear, shift_activity)
    if coast:
      # Coast supplies no gas-response samples; retain only a valid learned gas reduction.
      offset = self.integral
      if drivetrain != self.drivetrain or offset >= 0.0:
        self.reset()
      else:
        self.clear_observer()
        self.integral = offset
        self.paused = True
      return 0.0
    if self.paused:
      if drivetrain != self.drivetrain:
        self.reset()
      else:
        # This offset was not issued during coast; rebase it to the current gas headroom.
        self.correction = float(np.clip(self.integral, *limits))
        self.paused = False
    if drivetrain != self.drivetrain:
      self.clear_observer()
    elif self.pitch_peak is not None and pitch < self.pitch_peak - 0.01:
      # Falling grade can invalidate added gas, but should not erase overacceleration feedback.
      if self.integral > 0.0:
        self.clear_observer()
      self.pitch_peak = pitch
    self.drivetrain = drivetrain
    self.pitch_peak = pitch if self.pitch_peak is None else max(self.pitch_peak, pitch)
    self.requests.append((request, cap_weight))
    # Fresh feedback and warmup use the mapped headroom.
    lower = limits[0]
    upper = limits[1]
    target = 0.0
    if len(self.requests) > ODYSSEY_RESPONSE_DELAY_FRAMES:
      delayed_request, delayed_cap_weight = self.requests[0]
      response_gain = ODYSSEY_RESPONSE_COUNTS_PER_ACCEL + (ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL -
                                                          ODYSSEY_RESPONSE_COUNTS_PER_ACCEL) * float(np.clip(delayed_cap_weight, 0.0, 1.0))
      residual = self.error.update(delayed_request - aego)
      # Delayed feedback must agree with the latest acceleration error.
      residual = float(np.clip(residual, min(request - aego, 0.0), max(request - aego, 0.0)))
      proportional = response_gain * residual
      integral = float(np.clip(self.integral + ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * residual * DT_CTRL * 2,
                                lower, upper))
      target = proportional + integral
      # Retain the offset at zero error without accumulating beyond usable gas authority.
      if lower <= target <= upper or (target > upper and residual < 0.0) or (target < lower and residual > 0.0):
        self.integral = integral
      target = float(np.clip(proportional + self.integral, lower, upper))
      if target > 0.0 and residual < -ODYSSEY_COAST_RELEASE_ERROR:
        # Withdraw added gas on qualified excess without discarding the learned offset.
        target = min(target, max(0.0, self.correction + proportional))
    else:
      # Observed excess acceleration can reduce gas before the delayed observer is ready.
      proportional = float(np.clip(ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * min(request - aego, 0.0), lower, 0.0))
      target = float(np.clip(self.correction, lower, upper))
      # Scale withdrawal of an issued correction to the current reverse error.
      if (request - aego) * target < 0.0:
        target = float(np.clip(target + ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * (request - aego), lower, max(target, 0.0)))
      if request < aego:
        target = min(target, max(self.correction, 0.0) + proportional)
      # Seed the offset after accounting for the current proportional reduction.
      self.integral = target - proportional
    self.correction = float(np.clip(target, self.correction - ODYSSEY_RESPONSE_SLEW_COUNTS,
                                     self.correction + ODYSSEY_RESPONSE_SLEW_COUNTS))
    return self.correction


def odyssey_brake_release(now_nanos, request, aego, previous_brake, braking_ts_nanos, car_gas, gear, eligible, passive_accel=None,
                          coast_from_brake=False):
  """Prefer qualified coast when it can supply the requested deceleration."""
  if (not eligible or passive_accel is None or not all(math.isfinite(v) for v in (request, aego, car_gas, gear, passive_accel)) or
      not 1 <= gear <= 10 or car_gas != 0 or
      braking_ts_nanos <= 0 or not 0 <= now_nanos - braking_ts_nanos < 60_000_000):
    return False
  # Require excess braking to release; tolerate the existing response margin for a qualified handoff.
  margin = -ODYSSEY_COAST_RELEASE_ERROR if previous_brake else ODYSSEY_COAST_RELEASE_ERROR
  forecast_margin = ODYSSEY_COAST_RELEASE_ERROR if coast_from_brake and aego <= request else 0.0
  return request < 0.0 and passive_accel <= request + forecast_margin and aego < request + margin


class OdysseyCoastResponse:
  """Return a prior passive estimate, or None if unlearned or currently ineligible."""
  def __init__(self):
    self.drag = {}
    self.samples = deque(maxlen=5)
    self.gear = None
    self.coast_start = None

  def reset(self):
    self.__init__()

  def update(self, frame, gear, pitch, aego, coast, eligible):
    if not all(math.isfinite(v) for v in (gear, pitch, aego)):
      self.reset()
      return None
    if not eligible or not 1 <= gear <= 10:
      # Idle stop pauses forward-gear observations; keep completed calibration.
      self.gear = None
      self.coast_start = None
      self.samples.clear()
      return None
    if gear != self.gear:
      self.gear = gear
      self.coast_start = None

    gravity = -ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    forecast = gravity + self.drag[gear] if gear in self.drag else None
    if not coast:
      self.coast_start = None
    else:
      if self.coast_start is None:
        self.coast_start = frame
        # Median observations belong to one uninterrupted coast.
        self.samples.clear()
      # Acquire the first five settled samples promptly; retain the learned update cadence.
      if frame - self.coast_start >= 50 and (gear not in self.drag or frame % 10 == 0):
        self.samples.append(aego - gravity)
        if len(self.samples) == self.samples.maxlen:
          self.drag[gear] = sorted(self.samples)[2]
    return forecast


class OdysseyGasCoastRelease:
  """Release excess gas into coast using prior coast response or qualified gas exhaustion."""
  def __init__(self):
    self.requests = deque(maxlen=12)
    self.gear = None
    self.active = False
    self.exhausted = False

  def reset(self):
    self.__init__()

  def update(self, now_nanos, request, aego, speed, pitch, gear, passive_accel, previous_gas, eligible, gas_floor=False,
             floor_excess=False, downhill=True):
    exhausted_gas = previous_gas and gas_floor and floor_excess
    if (not eligible or (not downhill and not exhausted_gas and not self.exhausted) or not 1 <= gear <= 10 or
        not all(math.isfinite(v) for v in (request, aego, speed, pitch)) or
        (passive_accel is not None and not math.isfinite(passive_accel))):
      self.reset()
      return False
    if gear != self.gear:
      self.reset()
      self.gear = gear
    self.requests.append((now_nanos, request))
    prior = next((sample for sample in self.requests if sample[0] <= now_nanos - ODYSSEY_REQUEST_LOOKBACK_NS), None)
    request_slope = (request - prior[1]) / ((now_nanos - prior[0]) * 1e-9) if prior is not None else 0.0

    downhill_gravity = -ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    # A prior forecast must not undo exhausted-gas coast while measured excess remains.
    if (request <= ODYSSEY_ROAD_BRAKE_ENTRY or
        (passive_accel is not None and passive_accel - request < 0.05 and
         not self.exhausted) or
        not ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= speed <= 35.0 or
        (downhill_gravity <= ODYSSEY_COAST_RELEASE_GRAVITY and not self.exhausted) or
        (aego - request < 0.05 and not
         (passive_accel is not None and passive_accel - request >= 0.05 and
          aego >= request - ODYSSEY_COAST_RELEASE_ERROR))):
      self.active = self.exhausted = False
    if (not self.active and (previous_gas or passive_accel is not None) and ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= speed <= 35.0 and
        ODYSSEY_ROAD_BRAKE_ENTRY < request and
        aego - request > ODYSSEY_COAST_RELEASE_ERROR and
        (exhausted_gas or
         (downhill and downhill_gravity > ODYSSEY_COAST_RELEASE_GRAVITY and
          ((passive_accel is not None and passive_accel - request > ODYSSEY_COAST_RELEASE_ERROR) or
           (passive_accel is None and request < 0.0 and gas_floor and request_slope < -0.20))))):
      self.active = True
      self.exhausted = exhausted_gas
    return self.active


class OdysseyLearnerState:
  """The learner's own controller state (his CarController attributes), kept apart from today's controller."""
  def __init__(self):
    self.gas_response = OdysseyGasResponse()
    self.coast_response = OdysseyCoastResponse()
    self.gas_coast_release = OdysseyGasCoastRelease()
    self.brake_response_error = FirstOrderFilter(0.0, ODYSSEY_RESPONSE_ERROR_TAU, DT_CTRL * 2)

  def disengage(self):
    """His not-longActive reset (the brake/gas selection flags live on the controller and are reset there)."""
    self.gas_response.reset()
    self.brake_response_error.x = 0.0
    self.gas_coast_release.reset()


def odyssey_learner_long(cc, learner, CC, CS, accel, now_nanos, odyssey_pitch_valid, previous_gas_command, previous_accel,
                         load_factor):
  """brikowski c91503afe, CarController.update, Odyssey branch (gas and brake domain selection, coast and gas learning).

  cc is our CarController: it holds the shared selection state (odyssey_gas_selected, odyssey_brake_selected,
  odyssey_brake_start_frame, odyssey_gas_bridge_active, odyssey_pitch, accel, gas, frame, params). cc.gas holds this
  frame's gas map command on entry. Returns (gas_domain, brake_domain) for create_acc_commands.
  """
  actuators = CC.actuators
  previous_gas = cc.odyssey_gas_selected
  # Inactive brake feedback represents passive response only after the command settles.
  settled_brake = ((cc.odyssey_brake_selected or learner.coast_response.gear == CS.odyssey_target_gear) and
                   cc.odyssey_brake_start_frame is not None and
                   cc.frame - cc.odyssey_brake_start_frame >= 2 * ODYSSEY_RESPONSE_DELAY_FRAMES)
  gear_state_valid = (CS.out.canValid and CS.odyssey_target_gear_ts_nanos > 0 and
                      0 <= now_nanos - CS.odyssey_target_gear_ts_nanos < 60_000_000)
  coast_state_valid = (gear_state_valid and
                       CS.odyssey_computer_braking_ts_nanos > 0 and
                       0 <= now_nanos - CS.odyssey_computer_braking_ts_nanos < 60_000_000)
  coast_inputs_valid = (coast_state_valid and odyssey_pitch_valid and
                        0 <= now_nanos - CS.odyssey_engine_torque_ts_nanos < 60_000_000)
  coast_observation_eligible = not CS.out.gasPressed and not CS.out.brakePressed and coast_inputs_valid
  coast_eligible = CC.longActive and actuators.longControlState == LongCtrlState.pid and coast_observation_eligible
  manual_coast_eligible = (not CC.longActive and 8.0 <= CS.out.vEgo <= 35.0 and
                           abs(CS.odyssey_user_brake) <= 0.015625 and coast_observation_eligible)
  if not coast_inputs_valid:
    learner.coast_response.reset()
  # Brake feedback can remain active when the computer-braking indication drops.
  brake_feedback_active = (CS.odyssey_computer_braking or
                           (math.isfinite(CS.odyssey_user_brake) and CS.odyssey_user_brake > 0.015625))
  if learner.coast_response.gear != CS.odyssey_target_gear:
    learner.brake_response_error.x = 0.0
  passive_accel = learner.coast_response.update(
    cc.frame, CS.odyssey_target_gear, CC.orientationNED[1] if odyssey_pitch_valid else 0.0, CS.out.aEgo,
    # A nonnegative brake hold is controlled response, not a passive coast observation.
    not previous_gas and (not cc.odyssey_brake_selected or
                          (settled_brake and accel < 0.0)) and abs(CS.odyssey_user_brake) <= 0.015625 and
    CS.odyssey_car_gas == 0.0 and
    not CS.odyssey_computer_braking, (coast_eligible or manual_coast_eligible) and coast_state_valid)
  response_excess = (learner.gas_response.drivetrain == (CS.odyssey_target_gear, CS.odyssey_shift_activity) and
                     learner.gas_response.error.x < -ODYSSEY_COAST_RELEASE_ERROR)
  # GAS_COMMAND has one-count resolution and the CAN packer rounds half up.
  release_gas = learner.gas_coast_release.update(
    now_nanos, accel, CS.out.aEgo, CS.out.vEgo, cc.odyssey_pitch.x,
    CS.odyssey_target_gear, passive_accel, previous_gas,
    coast_eligible and coast_state_valid, math.floor(previous_gas_command + 0.5) == 0,
    response_excess and CS.odyssey_car_gas == 0.0 and not CS.odyssey_computer_braking and abs(CS.odyssey_user_brake) <= 0.015625,
    odyssey_pitch_valid and CC.orientationNED[1] < 0.0 and cc.odyssey_pitch.x < 0.0)
  # A single wheel-acceleration dip must not override a settled coast forecast.
  brake_error = accel - CS.out.aEgo
  if (cc.odyssey_brake_selected and coast_eligible and coast_state_valid and
      passive_accel is not None and ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= CS.out.vEgo <= 35.0 and
      CS.odyssey_car_gas == 0.0 and math.isfinite(CS.odyssey_user_brake) and math.isfinite(brake_error)):
    learner.brake_response_error.update(brake_error)
  else:
    learner.brake_response_error.x = 0.0
  release_brake = odyssey_brake_release(
    now_nanos, accel, CS.out.aEgo, cc.odyssey_brake_selected, CS.odyssey_computer_braking_ts_nanos,
    CS.odyssey_car_gas, CS.odyssey_target_gear,
    coast_eligible and coast_state_valid and CS.out.vEgo >= ODYSSEY_LOW_SPEED_DOMAIN_VEGO and
    (not cc.odyssey_brake_selected or brake_feedback_active),
    passive_accel, settled_brake and accel >= previous_accel and
    (not cc.odyssey_brake_selected or learner.brake_response_error.x > ODYSSEY_COAST_RELEASE_ERROR))
  release_brake |= (settled_brake and accel > 0.0 and
                    brake_error > ODYSSEY_COAST_RELEASE_ERROR and
                    learner.brake_response_error.x > ODYSSEY_COAST_RELEASE_ERROR)
  gas_accel = accel
  cap_weight = 0.0
  bridge_weight = 0.0
  if (odyssey_pitch_valid and CC.orientationNED[1] * cc.odyssey_pitch.x > 0.0 and
      actuators.longControlState == LongCtrlState.pid and not CS.out.gasPressed):
    gas_accel = odyssey_grade_gas_accel(accel, cc.odyssey_pitch.x)
    bridge_weight = odyssey_grade_weight(cc.odyssey_pitch.x)
  if actuators.longControlState == LongCtrlState.pid and not CS.out.gasPressed:
    cc.gas, cap_weight = odyssey_gas_feedforward(accel, gas_accel, CS.out.vEgo, cc.params, load_factor)
  passive_response_eligible = ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= CS.out.vEgo <= 35.0 and passive_accel is not None
  coast_start = learner.coast_response.coast_start
  settled_coast = (not cc.odyssey_brake_selected and coast_start is not None and
                   cc.frame - coast_start >= 2 * ODYSSEY_RESPONSE_DELAY_FRAMES)
  response_gas = (passive_response_eligible and accel < 0.0 and passive_accel < accel and
                  not CS.odyssey_computer_braking and
                  ((previous_gas and (CS.out.aEgo <= accel + ODYSSEY_COAST_RELEASE_ERROR or
                                      (not response_excess and not cc.odyssey_gas_bridge_active))) or
                   (passive_accel < accel - ODYSSEY_COAST_RELEASE_ERROR and
                    CS.odyssey_car_gas == 0.0 and CS.out.aEgo < accel - ODYSSEY_COAST_RELEASE_ERROR and
                    (settled_coast or settled_brake))))
  # Below the raw gas split, calibrated gas needs qualified live response for continuity.
  response_gas |= (previous_gas and not cc.odyssey_gas_bridge_active and
                   coast_eligible and coast_state_valid and not brake_feedback_active and
                   ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= CS.out.vEgo <= 35.0 and accel < 0.0 and
                   (passive_accel is None or passive_accel < accel) and
                   (learner.gas_response.correction > 0.0 or
                    gas_accel > ODYSSEY_GAS_MAP_FLOOR) and
                   CS.out.aEgo <= accel + ODYSSEY_COAST_RELEASE_ERROR)
  if response_gas:
    gas_selected, brake_selected = True, False
    cc.odyssey_gas_bridge_active = False
  # A prior coast forecast and fresh response can establish brake need before the learning dwell.
  elif (passive_response_eligible and not previous_gas and not cc.odyssey_brake_selected and
        coast_start is not None and passive_accel > accel + ODYSSEY_COAST_RELEASE_ERROR and
        CS.out.aEgo > accel + ODYSSEY_COAST_RELEASE_ERROR):
    gas_selected, brake_selected = False, True
  else:
    gas_selected, brake_selected = odyssey_command_domains(accel, CS.out.vEgo,
                                                            cc.odyssey_brake_selected,
                                                            previous_gas,
                                                            cc.odyssey_gas_bridge_active,
                                                            gas_accel, release_brake, release_gas,
                                                            passive_accel if passive_response_eligible else None)
  # A released brake command can precede the received brake feedback returning to idle.
  waiting_for_brake_release = (gas_selected and not previous_gas and coast_eligible and coast_state_valid and
                               ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= CS.out.vEgo <= 35.0 and brake_feedback_active)
  if waiting_for_brake_release:
    gas_selected = False
  # Keep settled brake history through qualified coast and the feedback wait.
  if brake_selected and not cc.odyssey_brake_selected:
    cc.odyssey_brake_start_frame = cc.frame
  elif not brake_selected and not (waiting_for_brake_release or (release_brake and settled_brake and not gas_selected)):
    cc.odyssey_brake_start_frame = None
  cc.odyssey_brake_selected = CC.longActive and brake_selected
  cc.odyssey_gas_selected = gas_selected
  # Honda brake holds do not measure gas-only response.
  feedback_state_valid = (CC.longActive and not cc.odyssey_gas_bridge_active and odyssey_pitch_valid and
                          actuators.longControlState == LongCtrlState.pid and
                          not CS.out.gasPressed and not CS.out.brakePressed and
                          coast_state_valid and math.isfinite(CS.odyssey_user_brake) and not brake_feedback_active)
  feedback_eligible = (feedback_state_valid and gas_selected and
                       (previous_gas or (accel >= 0.0 and learner.gas_response.integral < 0.0)) and
                       (gas_accel > ODYSSEY_GAS_MAP_FLOOR or response_gas))
  correction = learner.gas_response.update(accel, CS.out.aEgo, cc.odyssey_pitch.x,
                                           CS.out.vEgo, CS.odyssey_target_gear, feedback_eligible,
                                           cap_weight, CS.odyssey_shift_activity,
                                           limits=(-cc.gas, cc.params.BOSCH_GAS_LOOKUP_V[-1] - cc.gas),
                                           coast=feedback_state_valid and not gas_selected and not brake_selected)
  if feedback_eligible:
    cc.gas = float(np.clip(cc.gas + correction, 0.0, cc.params.BOSCH_GAS_LOOKUP_V[-1]))
  if CC.longActive and brake_selected and not CS.out.gasPressed and not CS.out.brakePressed:
    cc.accel = odyssey_creep_brake_accel(cc.accel, CS.out.vEgo)
  # A qualified passive-response shortfall can use gas and existing feedback at the map floor.
  cc.gas, cc.odyssey_gas_bridge_active = odyssey_gas_command(accel, cc.gas, gas_selected,
                                                             previous_gas or response_gas, cc.odyssey_gas_bridge_active,
                                                             bridge_weight)
  return gas_selected, brake_selected
