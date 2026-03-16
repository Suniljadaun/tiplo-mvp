"""
exercise_classifier.py - Tiplo Exercise Classifier
Detects: Squat | Arm Raise | Bird Dog | Deep Lunge | Back Extension

Stability features:
- Angle smoothing over rolling window
- Hysteresis thresholds (different enter/exit values)
- Confidence scoring per landmark visibility
- Orientation detection with margin
"""

# 'import' syntax to bring in the external 'numpy' library and bind it 'as' the alias 'np'. Semantically, this provides numerical computing functions.
import numpy as np
# 'from' ... 'import' syntax to load just the 'deque' class from the 'collections' module. Semantically, this is a fast double-ended queue.
from collections import deque
# 'from' ... 'import' syntax reading from our local 'utils.angles' package to get the 'calculate_angle' function. Semantically, it calculates joint angles.
from utils.angles import calculate_angle

# ── Landmark indices ─────────────────────────────────────────────────────────
# Assigning the integer 11 to L_SHOULDER and 12 to R_SHOULDER using tuple unpacking syntax. Semantically maps MediaPipe ids for shoulders.
L_SHOULDER, R_SHOULDER = 11, 12
# Assigning 13 to L_ELBOW and 14 to R_ELBOW using tuple unpacking. Maps elbows.
L_ELBOW,    R_ELBOW    = 13, 14
# Assigning 15 to L_WRIST and 16 to R_WRIST using tuple unpacking. Maps wrists.
L_WRIST,    R_WRIST    = 15, 16
# Assigning 23 to L_HIP and 24 to R_HIP using tuple unpacking. Maps hips.
L_HIP,      R_HIP      = 23, 24
# Assigning 25 to L_KNEE and 26 to R_KNEE using tuple unpacking. Maps knees.
L_KNEE,     R_KNEE     = 25, 26
# Assigning 27 to L_ANKLE and 28 to R_ANKLE using tuple unpacking. Maps ankles.
L_ANKLE,    R_ANKLE    = 27, 28

# ── Per-joint rolling smoothers ──────────────────────────────────────────────
# Syntax for creating an empty dictionary '{}' mapped to the variable '_smoothers'. Semantically stores history of angle values.
_smoothers = {}

# Define a function 'smooth' taking a 'key', a 'value', and a default integer parameter 'window=6'. Semantically averages last 'window' values.
def smooth(key, value, window=6):
    # Syntax 'if' statement checking if 'value' is exactly the 'None' singleton. Semantically, missing data stops smoothing logic.
    if value is None:
        # Return syntax passing 'None' back to the caller.
        return None
    # Syntax 'if' statement with 'not in' operator checking dictionary keys. Semantically checks if we have a deque for this joint yet.
    if key not in _smoothers:
        # Syntax dynamically adding a new 'deque' to the dictionary at 'key' with max size 'window'. Semantically initializes the queue.
        _smoothers[key] = deque(maxlen=window)
    # Target the specific deque via key and call its 'append()' method. Semantically pushes the new angle frame into the rolling window.
    _smoothers[key].append(value)
    # Syntax: np.mean() computes average, float() casts it to native python float, return passes it back. Semantically returns smoothed angle.
    return float(np.mean(_smoothers[key]))

# ── Helpers ──────────────────────────────────────────────────────────────────
# Function definition syntax with a default argument 'threshold=0.45'. Semantically returns boolean true if landmark is safely visible.
def _vis(lm, idx, threshold=0.45):
    # Syntax: indexing list 'lm' with 'idx', accessing attribute 'visibility', evaluating '>', returning. Semantically compares detection confidence.
    return lm[idx].visibility > threshold

# Function to get the y-coordinate. Syntax: standard def returning property. Semantically returns vertical position.
def _y(lm, idx):  return lm[idx].y
# Function to get the x-coordinate. Syntax: standard def returning property. Semantically returns horizontal position.
def _x(lm, idx):  return lm[idx].x
# Function returning a tuple. Syntax: returns (x, y). Semantically abstracts gathering a 2D point from the landmark array.
def _pt(lm, idx): return (lm[idx].x, lm[idx].y)

# Function definition using "*indices" syntax indicating an arbitrary number of positional arguments. Semantically averages Y coordinates.
def _avg_y(lm, *indices):
    # List comprehension computing a list of 'y' values optionally filtering only visibly confident landmarks via 'if _vis()'.
    vals = [_y(lm, i) for i in indices if _vis(lm, i)]
    # Conditional (ternary-like logic via if/else). Syntax: float(np.mean()) if list has items else None. Semantically returns average Y or None.
    return float(np.mean(vals)) if vals else None

# Function taking arbitrary indices. Semantically calculates the average confidence score across several joints.
def _confidence(lm, *indices):
    # Syntax: List comprehension inside np.mean(), converted to float. Gathers and averages 'visibility' attributes.
    return float(np.mean([lm[i].visibility for i in indices]))

# Function definition to get angle of a knee. Syntax: takes landmark list 'lm' and string 'side'.
def _knee_angle(lm, side):
    # Syntax: standard docstring. Documents function behavior.
    """Returns smoothed knee angle for given side, or None."""
    # Syntax: string equality check '=='. Semantically checks if we are evaluating the left side.
    if side == 'L':
        # Tuple unpacking dynamically assigning the logical Left joints.
        hp, kn, ank = L_HIP, L_KNEE, L_ANKLE
    # Syntax: else block for all other conditions (i.e. 'R').
    else:
        # Tuple unpacking dynamically assigning the logical Right joints.
        hp, kn, ank = R_HIP, R_KNEE, R_ANKLE

    # Syntax: chained logical 'and' operators calling function '_vis()'. Semantically ensures all 3 points forming the knee angle are visible.
    if _vis(lm, hp) and _vis(lm, kn) and _vis(lm, ank):
        # Syntax: calling 'calculate_angle' feeding it tuples from '_pt()'. Semantically computes the angle between Hip -> Knee -> Ankle.
        a = calculate_angle(_pt(lm, hp), _pt(lm, kn), _pt(lm, ank))
        # Syntax: formatted string (f-string) as key, calling 'smooth', returning. Semantically applies running average and returns the angle.
        return smooth(f'knee_{side}', a)

    # Ankle not visible — fallback to hip angle (shoulder→hip→knee)
    # Syntax: inline ternary expression checking 'side' to assign appropriate shoulder constant.
    sh = L_SHOULDER if side == 'L' else R_SHOULDER
    # Syntax: chained check. Semantically tests visibility of the alternate joint sequence for fallback logic.
    if _vis(lm, sh) and _vis(lm, hp) and _vis(lm, kn):
        # Syntax: assigning calculation result to variable 'a'. Semantically computes angle between Shoulder -> Hip -> Knee.
        a = calculate_angle(_pt(lm, sh), _pt(lm, hp), _pt(lm, kn))
        # Syntax: formatted string key, calling 'smooth()'.
        return smooth(f'hip_{side}', a)

    # Syntax: return singleton. Semantically, if points are missing, return nothing.
    return None


# ── Body orientation ─────────────────────────────────────────────────────────
# Syntax: defining function parsing 'lm'. Semantically deduces the overall posture state to filter possible exercises.
def get_body_orientation(lm):
    # Syntax docstring mapping expected return strings.
    """Returns: 'upright' | 'all_fours' | 'prone' | 'unknown'"""
    # Syntax: assigning the returned average Y. Semantically finds horizontal center point of the shoulders.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    # Syntax: assigning returned average Y. Semantically finds horizontal center point of the hips.
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    # Syntax: checking if "is None", combined by 'or'. Semantically validates that both key upper/lower body segments are available.
    if sh_y is None or hp_y is None:
        # Syntax: string return. Semantically yields 'unknown' state if body parts are occluded.
        return 'unknown'

    # Syntax: numerical subtraction. Semantically, since Y goes down (0 top, 1 bottom), positive means hips are physically below shoulders.
    diff = hp_y - sh_y  # positive = hips below shoulders

    # Syntax: greater-than check. Semantically, if hips are notably below shoulders, person is standing/squatting.
    if diff > 0.22:
        # Returns upright string token.
        return 'upright'
    # Syntax: strictly checking absolute value. Semantically, if the vertical shoulder/hip distances are close, body is horizontal.
    if abs(diff) <= 0.22:
        # Syntax inline ternary return. Semantically, if hips are high up (low Y coordinate), person is on all fours; if lower, lying prone entirely.
        return 'all_fours' if hp_y < 0.60 else 'prone'
    # Syntax: default fallback return.
    return 'unknown'


# ── EXERCISE 1: Squat ────────────────────────────────────────────────────────
# Hysteresis: enter active below 130°, exit above 155°
# Symmetry check: both knees must bend within 25° of each other
# Syntax: Int assignment setting the threshold angle parameter at which a squat conceptually 'starts'.
SQUAT_ENTER    = 130
# Syntax: Int assignment setting the threshold angle at which a squat conceptually 'ends' or rests.
SQUAT_EXIT     = 155
# Syntax: Int assignment defining acceptable degree difference between two knees to qualify as symmetric.
SQUAT_SYMMETRY = 25   # max allowed angle difference between knees

# Syntax: Boolean assignment starting false. Semantically tracks state if user is currently inside a squat movement.
_squat_active = False

# Syntax: function definition extracting state tuple.
def detect_squat(lm):
    # Syntax: docstring block.
    """
    Squat: upright, both knees bend symmetrically.
    Distinguished from lunge by symmetry of both knee angles.
    Returns: (phase, avg_angle, debug)
    """
    # Syntax: global keyword telling Python to modify the module-scoped `_squat_active` rather than assigning locally.
    global _squat_active

    # Syntax: variable assignment from function return.
    orientation = get_body_orientation(lm)
    # Syntax: inequality string comparison. Semantically ensures squat only registers when standing.
    if orientation != 'upright':
        # Syntax: triple return. Semantically yields 'wrong_position', null data, and formatted debug explanation string.
        return 'wrong_position', None, f"orient:{orientation} — stand up"

    # Syntax: invoking knee helper. Retrieves left knee angle.
    l_angle = _knee_angle(lm, 'L')
    # Syntax: retrieving right knee angle.
    r_angle = _knee_angle(lm, 'R')

    # Syntax: logical 'and', 'is None' checks. Validates if we have any valid knee angle calculations.
    if l_angle is None and r_angle is None:
        # Immediate abort return if legs are completely hidden.
        return None, None, "no knee landmarks"

    # Use whichever sides we have
    # Syntax: List comprehension filtering non-None angles. Semantically creates array of good knee readings.
    available = [a for a in [l_angle, r_angle] if a is not None]
    # Syntax: invoking numpy's mean over the valid array and converting to float. Computes average knee bend.
    avg_angle = float(np.mean(available))

    # Symmetry check — if only one side visible, skip symmetry
    # Syntax: Boolean defaulting to True.
    is_symmetric = True
    # Syntax: Implicit Truthy check. Ensures both knees are rendering before measuring their difference.
    if l_angle and r_angle:
        # Syntax: absolute math function compared to int threshold. Determines if the squat matches standard symmetry.
        is_symmetric = abs(l_angle - r_angle) < SQUAT_SYMMETRY

    # Syntax: f-string using nested ternary formatting (f"{var if var else '--'}"). Semantically creates log string.
    debug = (f"L:{f'{l_angle:.0f}' if l_angle else '--'} "
             f"R:{f'{r_angle:.0f}' if r_angle else '--'} "
             f"avg:{avg_angle:.0f} sym:{is_symmetric}")

    # Hip drop check
    # Syntax: Fetch average normalized height of user's shoulders.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    # Syntax: Fetch average normalized height of user's hips.
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    # Syntax: subtraction resolving absolute height variance if variables aren't None using ternary conditional.
    hip_drop = (hp_y - sh_y) if (sh_y and hp_y) else 0

    # Hysteresis
    # Syntax: complex conditional 'not' gate, float bounds, booleans, and float magnitude check.
    # Semantically: We enter 'active' phase if not already in it AND the knee average is below threshold AND the stance is symmetric AND the hips have sufficiently dropped.
    if not _squat_active and avg_angle < SQUAT_ENTER and is_symmetric and hip_drop > 0.20:
        # State toggle changing global state variable to True.
        _squat_active = True
    # Syntax: 'elif' fallback testing 'active' condition combined with angle exit threshold check.
    elif _squat_active and avg_angle > SQUAT_EXIT:
        # Resets global boolean state to exit current repetition cleanly.
        _squat_active = False

    # Syntax: conditional branching rendering the final resulting posture phase strings.
    if _squat_active:
        return 'active', avg_angle, debug
    # Syntax: evaluates if the individual is partially bending but not actively "in" or "out" of the rep frame.
    elif avg_angle < SQUAT_EXIT:
        return 'partial', avg_angle, debug
    # Syntax: standard fallback returning resting phase info strings.
    else:
        return 'rest', avg_angle, debug


# ── EXERCISE 2: Arm Raise ────────────────────────────────────────────────────
# Active: wrist(s) rise above shoulder level
# Hysteresis: enter when wrist_y < shoulder_y - 0.06, exit when wrist_y > shoulder_y + 0.02
# Syntax: Float assignment parameter for minimum raise distance above shoulders.
ARM_ENTER_MARGIN = 0.06   # wrist must be this much above shoulder
# Syntax: Float assignment establishing the fallback exit threshold limit.
ARM_EXIT_MARGIN  = 0.02   # wrist falls back to shoulder level + margin

# Syntax: Defining global state tracker variable returning boolean defaults.
_arm_active = False

# Function parsing landmark data returning phase logic for arm raising.
def detect_arm_raise(lm):
    # Syntax: docstring. Note logic flow details.
    """
    Arm Raise: standing, one or both wrists rise above shoulder height.
    Returns: (phase, side, debug)
    """
    # Exposing the outer tracking boolean via 'global' scoping wrapper keyword syntax.
    global _arm_active

    # Pull user posture orientation using helper method binding to parameter matching.
    orientation = get_body_orientation(lm)
    # Syntax: If not matching hardcoded standard upright context string, terminate loop iteration block via multiple tuple return.
    if orientation != 'upright':
        return 'wrong_position', None, f"orient:{orientation} — stand up"

    # Gets vertical median height (y value) of detected upper shoulders if seen.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    # Rejects loop state processing returning None blocks entirely.
    if sh_y is None:
        return None, None, "shoulders not visible"

    # Calculating dynamic float limits specific to this current snapshot via normalized mathematical constraints.
    enter_thresh = sh_y - ARM_ENTER_MARGIN
    exit_thresh  = sh_y + ARM_EXIT_MARGIN

    # Computing Boolean logic using visible constraints + numerical limit checks for both extreme appendages.
    # Semantically checks if the left hand is successfully raised.
    l_raised = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < enter_thresh
    # Semantically checks if right hand is physically elevated above bound parameter.
    r_raised = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < enter_thresh
    # Confirms hand is formally dropped downward checking out-of-bounds or non-visible failure states.
    l_down   = not _vis(lm, L_WRIST) or _y(lm, L_WRIST) > exit_thresh
    r_down   = not _vis(lm, R_WRIST) or _y(lm, R_WRIST) > exit_thresh

    # Defining verbose structural formatting mapping variable metrics internally tracking debug values into formatted f-string constants.
    debug = (f"sh_y:{sh_y:.2f} thresh:{enter_thresh:.2f} "
             f"L_wrist:{_y(lm,L_WRIST):.2f} R_wrist:{_y(lm,R_WRIST):.2f} "
             f"L_raised:{l_raised} R_raised:{r_raised}")

    # Determine side
    # Syntax: 'if' testing multiple booleans mapped via 'and'. Semantically maps 'both' strings if multi-limb engaged.
    if l_raised and r_raised:
        side = 'both'
    # Syntax: 'elif' mapped isolated bool check. Semantically denotes "left" tag strings.
    elif l_raised:
        side = 'left'
    # Fallback to string mapped right check.
    elif r_raised:
        side = 'right'
    # Syntax fallback assigning None object when unidentifiable contexts occur.
    else:
        side = None

    # Hysteresis
    # Applies enter thresholds mapping standard state changes using logically grouped conditionals against boolean values and OR groupings wrapped via brackets.
    if not _arm_active and (l_raised or r_raised):
        _arm_active = True
    # Evaluates combined exit constraint mappings using paired conjunction logical mapping AND boolean gates.
    elif _arm_active and l_down and r_down:
        _arm_active = False

    # Standard explicit block validation triggering variable structural output formats via conditional tuple parsing.
    if _arm_active:
        return 'active', side, debug
    else:
        return 'rest', None, debug


# ── EXERCISE 3: Bird Dog ─────────────────────────────────────────────────────
# Syntax: Module-level boolean state var. Semantically registers active tracking memory flags.
_bird_dog_active = False

# Routine def handling "bird dog" detection. Core goal: identify quadruped movements matching explicit constraints.
def detect_bird_dog(lm):
    """
    On all fours, extend opposite arm + leg simultaneously.
    Returns: (phase, side, confidence, debug)
    """
    # Exposing the global var. Validating namespace accessibility using "global" literal.
    global _bird_dog_active

    # Determine pose posture alignment calling modular orientation helper algorithm.
    orientation = get_body_orientation(lm)
    # Syntax check mapping strict constraints requiring non-upright mapping arrays using multiple exclusion mapping sequences ('not in').
    if orientation not in ('all_fours', 'prone'):
        return 'wrong_position', None, 0.0, f"orient:{orientation} — get on hands+knees"

    # Assigning float variables containing horizontal aligned metric mappings isolating joints if properly returned via visible function structures.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    # If missing null validation returning string formatted aborts bypassing remainder blocks explicitly.
    if sh_y is None or hp_y is None:
        return None, None, 0.0, "landmarks missing"

    # Static assignment representing expected vertical elevation ranges mapping minimum required positional height differentials.
    arm_thresh = sh_y - 0.07
    leg_thresh = 0.14

    # Variable boolean evaluation comparing left & right limbs using < bounds checks ensuring wrist elevation over minimal thresh.
    l_arm_ext = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < arm_thresh
    r_arm_ext = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < arm_thresh
    # Leg processing evaluates ankle relative delta to hip checking minimal deviations indicating active lifts mapping 'abs()'.
    l_leg_ext = _vis(lm, L_ANKLE) and abs(_y(lm, L_ANKLE) - hp_y) < leg_thresh
    r_leg_ext = _vis(lm, R_ANKLE) and abs(_y(lm, R_ANKLE) - hp_y) < leg_thresh

    # Mapping variables dynamically mapping boolean evaluations logging via textual interpolation mapping.
    debug = (f"L_arm:{l_arm_ext} R_arm:{r_arm_ext} "
             f"L_leg:{l_leg_ext} R_leg:{r_leg_ext}")

    # Evaluating composite boolean states indicating if opposite limb configurations trigger passing metrics.
    right_arm_left_leg = r_arm_ext and l_leg_ext
    left_arm_right_leg = l_arm_ext and r_leg_ext
    # Evaluating generic activity checking loose any limb extension states using chain 'or' mapping literals logic gates.
    any_extension      = l_arm_ext or r_arm_ext or l_leg_ext or r_leg_ext

    # Aggregating visibility confidence metrics across structural spine elements mapping float scalar returns.
    conf = _confidence(lm, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)

    # Boolean mapping managing entry state mapping complex conjunctions crossing 'not' evaluations.
    if not _bird_dog_active and (right_arm_left_leg or left_arm_right_leg):
        _bird_dog_active = True
    # Evaluates boolean exits checking lack of all required activity states mapping simple var falsy conditions.
    elif _bird_dog_active and not any_extension:
        _bird_dog_active = False

    # Block return testing output formats parsing states. Returning strict type tuples mapped via variable literal assignments.
    if _bird_dog_active:
        # String mappings defining literal side descriptions formatted dynamically via conditional logic maps.
        side = 'R_arm+L_leg' if right_arm_left_leg else 'L_arm+R_leg'
        # Calling native Python 'min()' setting strict boundaries isolating scores wrapping numeric outputs >1.0 logic limits.
        return 'active', side, min(conf + 0.2, 1.0), debug
    elif any_extension:
        # Returning partial string tags representing isolated activity outputs filtering scaled mathematical outputs resolving float modifications.
        return 'partial', 'one_limb', conf * 0.6, debug
    else:
        # Fallback string mappings evaluating standard inactive rest phase blocks mapping standard tuple returns.
        return 'rest', None, conf, debug


# ── EXERCISE 4: Deep Lunge ───────────────────────────────────────────────────
# Constant assignments handling float and integer mappings mapping acceptable knee flexion bounds for lunges.
LUNGE_ENTER    = 115 # Syntax mapped representing degree threshold logic mapping active entering scopes.
LUNGE_EXIT     = 145 # Syntax representing release limits logging rest exit phases logically assigned int.
LUNGE_SYMMETRY = 30  # lunges are ASYMMETRIC — knees must differ by this much (syntax: inline int literal)

# Boolean logic mappings establishing dynamic global tracking reference objects.
_lunge_active = False

# Function declaration checking pose logic variables specific to single-leg lunging.
def detect_deep_lunge(lm):
    # Syntax text literal mapping docstring properties indicating logic flows evaluating functional returns explicitly.
    """
    One knee bends deeply while the other stays relatively straight.
    Key distinction from squat: asymmetric knee angles.
    Returns: (phase, side, angle, debug)
    """
    # Exposing the global var.
    global _lunge_active

    # Determine standing check using orientation helper variables handling constraints explicitly mapped over string equivalents.
    orientation = get_body_orientation(lm)
    if orientation != 'upright':
        # Reject blocks using conditional multi-var literal mapping formats formatting None data equivalents filtering bad parameters.
        return 'wrong_position', None, None, f"orient:{orientation} — stand up"

    # Evaluates isolated knee bends handling internal function mapping scopes explicitly routing strings identifying limbs.
    l_angle = _knee_angle(lm, 'L')
    r_angle = _knee_angle(lm, 'R')

    # Identifies edge case nullification where both variables equal None evaluating strict falsy states explicitly.
    if l_angle is None and r_angle is None:
        # Tuple sequence bypass mappings returning null values mapping log formats parsing exceptions.
        return None, None, None, "no knee landmarks"

    # Variable scope handling checking boolean existence mapping explicit conditionals parsing leg asymmetry rules sets.
    # Need both angles to confirm asymmetry
    if l_angle and r_angle:
        # Mathematical assignment identifying degree gap validating logic over float calculations ('abs' native func).
        asymmetry = abs(l_angle - r_angle)
        # Conditional string generation targeting explicit isolated elements finding minimum leg using less-than boolean maps.
        front_side  = 'L' if l_angle < r_angle else 'R'
        # Applies native min function assigning closest explicit element finding smallest int representing leading degree variables explicitly.
        front_angle = min(l_angle, r_angle)
    # Map mapping unconfirmed checks evaluating else fallback strings conditionally assigning literal bounds.
    else:
        # Only one side visible — can still detect (assigning strict fallback logic limit forcing 999 bypassing diff limit loops)
        asymmetry   = 999
        # Ternary conditional yielding L if it isn't falsy mapping literal character assigns evaluating string tags logically.
        front_side  = 'L' if l_angle else 'R'
        # Or variable assignment parsing truthy expressions mapping raw explicit scalar returns matching first valid variables.
        front_angle = l_angle or r_angle

    # Extract averages identifying user center-points using helper functions validating vertical shoulder variables globally.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    # Computing specific variables assigning explicitly filtered hip horizontal bounds over visibility logic map filtering blocks continuously.
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    # Conditional float math parsing scalar hip variance avoiding none errors handling literal map conditional parsing ternary validations conditionally.
    hip_drop = (hp_y - sh_y) if (sh_y and hp_y) else 0

    # Complex string representation using tuple map conditionals inside f-string string interpolation evaluating literal nested format string logic outputs formally.
    debug = (f"L:{f'{l_angle:.0f}' if l_angle else '--'} "
             f"R:{f'{r_angle:.0f}' if r_angle else '--'} "
             f"asym:{asymmetry:.0f} hip_drop:{hip_drop:.2f}")

    # Evaluating combined boolean evaluation checking threshold values separating logic states defining valid lunge states.
    is_lunge = front_angle < LUNGE_ENTER and (asymmetry > LUNGE_SYMMETRY or l_angle is None or r_angle is None)

    # Cross evaluating constraints linking specific 'not' variable gates mapping combined float conditions checking if state passes 0.25 literal map boundaries explicitly.
    if not _lunge_active and is_lunge and hip_drop > 0.25:
        _lunge_active = True
    # Evaluates boolean mapped checks handling greater-than int map tracking limits mapping exit threshold string mapping structures logically.
    elif _lunge_active and front_angle > LUNGE_EXIT:
        _lunge_active = False

    # Block formatting return arrays processing state variables explicitly defining phase conditionals mappings tracking float parameters explicitly via return arrays conditionally.
    if _lunge_active:
        return 'active', front_side, front_angle, debug
    elif front_angle < LUNGE_EXIT:
        return 'partial', front_side, front_angle, debug
    else:
        return 'rest', front_side, front_angle, debug


# ── EXERCISE 5: Back Extension ───────────────────────────────────────────────
# Assign variables to float literals creating fixed rules calculating body bounds defining structural elevation.
BACK_EXT_ENTER = 0.10
BACK_EXT_EXIT  = 0.05

# Setting global tracking booleans mapping string outputs managing phase structures mapping dynamic returns.
_back_ext_active = False

# Function declaration assigning procedural parameter parsing checking logic constraints for extension values.
def detect_back_extension(lm):
    # Syntax text literal mapping docstring properties indicating logic flows evaluating functional returns explicitly.
    """
    Prone, lift chest off floor.
    Returns: (phase, lift_amount, debug)
    """
    # Bringing global var inside function namespace manipulating strict state variables conditionally via literal assignments.
    global _back_ext_active

    # Check horizontal alignment via positional parsing tracking strings verifying literal map variables avoiding false positives mappings logically.
    orientation = get_body_orientation(lm)
    # Rejects standing alignments evaluating boolean map logic arrays filtering explicit 'not in' string maps.
    if orientation not in ('prone', 'all_fours'):
        # Exiting conditional block bypassing normal logic generating literal fallback values matching strict mapping constraints.
        return 'wrong_position', 0, f"orient:{orientation} — lie face down"

    # Evaluates horizontal constraints parsing functions generating explicit normalized location parameters conditionally explicitly filtering exceptions.
    sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
    hp_y = _avg_y(lm, L_HIP, R_HIP)
    # Conditional gate checking if variables equal None mapping tuple bypass formats parsing strings.
    if sh_y is None or hp_y is None:
        # Returns early avoiding syntax crashes evaluating float logic against none parameters explicitly generating exceptions natively.
        return None, 0, "landmarks missing"

    # Standard assignment map evaluating simple substraction generating tracking variances logging positional limits evaluating positional bounds directly.
    lift = hp_y - sh_y

    # Logging parameter interpolating literal map generation rounding metrics logging literal float scalars explicitly defining limits tracking variations dynamically.
    debug = f"sh_y:{sh_y:.2f} hp_y:{hp_y:.2f} lift:{lift:.3f}"

    # Conditional gates comparing variables parsing 'not' states combining limits crossing boundary mappings natively parsing true conditions.
    if not _back_ext_active and lift > BACK_EXT_ENTER:
        _back_ext_active = True
    # Evaluates combined exit constraint mappings using paired conjunction logical mapping AND boolean gates assigning boolean assignments.
    elif _back_ext_active and lift < BACK_EXT_EXIT:
        _back_ext_active = False

    # Block formatting return arrays processing state variables explicitly defining phase conditionals mappings tracking float parameters explicitly via return sequence formats dynamically.
    if _back_ext_active:
        return 'active', lift, debug
    elif lift > BACK_EXT_EXIT:
        return 'partial', lift, debug
    else:
        return 'rest', lift, debug


# ── Main classifier ──────────────────────────────────────────────────────────
# Main public function defining multi-condition map tracking limits dynamically calculating array elements optionally handling explicit values conditionally.
def classify_exercise(lm, target_exercise=None):
    # Determine base posture returning string tag checking horizontal mapping algorithms isolating standing vs ground alignments.
    orientation = get_body_orientation(lm)

    # ── Locked to specific exercise ───────────────────────────────────────
    # Evaluates strict argument matching evaluating boolean map string matching. Semantically limits bounds bypassing auto-detect functions checking parameters explicitly.
    if target_exercise == 'squat':
        # Tuple unpacking syntax calling local algorithm generating strict maps logging variables explicitly returning native limits arrays directly parsed dynamically.
        phase, angle, debug = detect_squat(lm)
        # Returns dictionary array matching json map parameters generating key-value pairs parsing variable mappings unconditionally resolving format logic internally.
        return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                'orientation': orientation, 'debug': debug}

    # Strict match 'if' syntax evaluating string mapping literal matching arm raise logic array constraints generating tracking limits tracking true conditionals matching explicitly.
    if target_exercise == 'arm_raise':
        # Unpacks explicit variables parsing custom algorithm conditionally finding positional maps dynamically validating logical inputs conditionally.
        phase, side, debug = detect_arm_raise(lm)
        # Generates JSON equivalent maps parsing array formatting structures capturing logic bounds mapping literal values unconditionally generating dictionaries linearly.
        return {'exercise': 'Arm Raise', 'phase': phase, 'side': side,
                'orientation': orientation, 'debug': debug}

    # Strict match 'if' syntax evaluating 'bird dog' conditionals natively finding mapping conditionals.
    if target_exercise == 'bird_dog':
        # Multi-variable mapping assigning four states generating strict outputs conditionally mapped via native array sequence limits.
        phase, side, conf, debug = detect_bird_dog(lm)
        # Parses explicit dictionary generation returning values conditionally matching object properties parsing internal bounds explicitly mapped inline dynamically.
        return {'exercise': 'Bird Dog', 'phase': phase, 'side': side,
                'confidence': conf, 'orientation': orientation, 'debug': debug}

    # Evaluates lunge explicitly matching inline strings explicitly checking mapping conditional bounds tracking variables directly validating mapped variables explicitly.
    if target_exercise == 'deep_lunge':
        # Four variable definitions matching returned limits unpacking tuple explicitly assigned variables conditionally.
        phase, side, angle, debug = detect_deep_lunge(lm)
        # Returns json object properties dictionary format dynamically assigned specific literal strings defining custom tags strictly mapping outputs implicitly identifying inputs.
        return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                'angle': angle, 'orientation': orientation, 'debug': debug}

    # Conditional map tracking float scalar limits matching explicit parameter mapping arrays conditionally validating logic outputs.
    if target_exercise == 'back_extension':
        # Function evaluation assigning phase arrays returning parameter blocks checking positional conditions strictly parsing tuple objects dynamically mapping outputs implicitly.
        phase, lift, debug = detect_back_extension(lm)
        # Dictionary map matching returned variables parsing string keys explicitly mapped isolating variables natively outputting dictionary structures conditionally.
        return {'exercise': 'Back Extension', 'phase': phase, 'lift': lift,
                'orientation': orientation, 'debug': debug}

    # ── Auto-detect by orientation ────────────────────────────────────────
    # Evaluates boolean limits checking standard map returns if target is None implicitly activating dynamic tracking unconditionally conditional checks handling state overrides.
    if orientation == 'upright':
        # Distinguish squat vs lunge vs arm raise (assigning explicit floats returning helper tracking variables)
        l_angle = _knee_angle(lm, 'L')
        r_angle = _knee_angle(lm, 'R')

        # Check arm raise first (can happen at any knee angle) calculating vertical float mapped limits.
        sh_y = _avg_y(lm, L_SHOULDER, R_SHOULDER)
        # Explicit evaluation tracking boolean variables checking existance resolving native logic limits.
        if sh_y:
            # Compound eval chaining not null limits with dynamic height comparison matching literal bounds explicitly assigned variables conditionally.
            l_raised = _vis(lm, L_WRIST) and _y(lm, L_WRIST) < sh_y - 0.06
            r_raised = _vis(lm, R_WRIST) and _y(lm, R_WRIST) < sh_y - 0.06
            # Implicit mapping tracking truthy states logging variables generating logical combinations conditionally.
            if l_raised or r_raised:
                # If variables parse TRUE evaluating literal object formats running internal mapped loops detecting arrays.
                phase, side, debug = detect_arm_raise(lm)
                # JSON block parsing return states escaping limits evaluating variables bypassing native loop limits isolating variables parsing string outputs.
                return {'exercise': 'Arm Raise', 'phase': phase, 'side': side,
                        'orientation': orientation, 'debug': debug}

        # Squat vs lunge: check symmetry conditionals mapping true constraints evaluating non-none conditions.
        if l_angle and r_angle:
            # Mathematical calculations assigning numerical bounds logging distance checking boolean maps explicitly conditionally mapping.
            asymmetry = abs(l_angle - r_angle)
            # Evaluates logical combinations comparing variable bounds to fixed integers mapping conditional mapping gates generating literal sequences conditionally tracking limits explicitly.
            if asymmetry > LUNGE_SYMMETRY and min(l_angle, r_angle) < LUNGE_EXIT:
                # Runs lunge conditionals matching asymmetric mappings mapping logical block states logging mapping variables explicitly implicitly parsing string values logically.
                phase, side, angle, debug = detect_deep_lunge(lm)
                # Dictionary map returning values escaping loop conditional variables parsing strict objects dynamically mapping implicit string properties isolating constraints natively mapping arrays.
                return {'exercise': 'Deep Lunge', 'phase': phase, 'side': side,
                        'angle': angle, 'orientation': orientation, 'debug': debug}
            # Fallback syntax handling else variables mapped isolating squat symmetry testing array outputs defining limits unconditionally tracking structures internally mapping dynamically.
            else:
                # Executes internal functions assigning tracking arrays defining logical outputs defining standard string mapping limits parsing constraints conditionally matching variables implicitly defining logic flows.
                phase, angle, debug = detect_squat(lm)
                # Default map tracking structures defining objects formatting dictionary properties returning values parsing conditions.
                return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                        'orientation': orientation, 'debug': debug}

        # Default upright = squat
        # Executes internal array outputs returning conditional values passing tuple structure variables bypassing array limits parsing output conditions logging state explicitly tracking float conditionals natively calculating variable conditions.
        phase, angle, debug = detect_squat(lm)
        # Default mapping tracking returns validating object limits checking variables logging output conditional values parsing dictionaries evaluating logical bounds linearly defining structure implicitly matching values actively mapping maps unconditionally parsing explicit logic formatting variables.
        return {'exercise': 'Squat', 'phase': phase, 'angle': angle,
                'orientation': orientation, 'debug': debug}

    # Conditional evaluating ground limits evaluating object types mapping exact string equivalence explicitly conditionally bypassing unassigned variables matching variables matching maps exactly checking conditions.
    if orientation == 'all_fours':
        # Resolves internal functions defining parameter structures matching logic outputs generating tracking limits evaluating variable combinations generating dynamic scalar conditions evaluating conditional formats actively.
        phase, side, conf, debug = detect_bird_dog(lm)
        # Returns formatted json checking object type variables defining sequence string limits checking mapping dictionary outputs evaluating limits capturing variables conditionally defining values dynamically outputting formatted dictionaries mapping explicitly validating parameters.
        return {'exercise': 'Bird Dog', 'phase': phase, 'side': side,
                'confidence': conf, 'orientation': orientation, 'debug': debug}

    # Evaluates boolean limits logging variables checking explicit mapping variables mapping strings tracking 'prone' state logging matching strict boolean limits generating valid limits mapping conditions parsing bounds linearly.
    if orientation == 'prone':
        # Evaluates variables unpacking internal arrays computing bounds logging dictionary variables checking states conditionally returning variables resolving functions logging values unconditionally generating variables conditionally identifying states explicitly matching dictionary structures natively defining strings implicitly checking logic conditional conditions.
        phase, lift, debug = detect_back_extension(lm)
        # Map assigning logical outputs generating variables mapping sequence dictionaries verifying limits parsing outputs evaluating variable lists dynamically mapping JSON conditionals implicitly validating float parameters explicitly returning conditionals testing maps visually mapping dictionaries.
        return {'exercise': 'Back Extension', 'phase': phase, 'lift': lift,
                'orientation': orientation, 'debug': debug}

    # Final fallback map parsing returns generating unassigned values testing parameters conditionally identifying arrays evaluating logic returning sequence JSON outputs unconditionally mapping default tracking variables processing variables generating dictionary literal representations conditional sequences.
    return {'exercise': 'Detecting...', 'phase': None,
            'orientation': orientation, 'debug': f'orient:{orientation}'}