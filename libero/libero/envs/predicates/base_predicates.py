from typing import List
import numpy as np
import robosuite.utils.transform_utils as T


class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()


class OnSide(UnaryAtomic):
    """
    Returns True if the object's local z-axis is approximately horizontal,
    i.e., the object is flipped onto one of its side faces. Optionally, we
    also require contact with the floor if available in the environment.

    Heuristic: |R[2,2]| < cos(theta_thresh), with theta_thresh ~ 60 deg.
    This excludes upright (|R[2,2]| ~ 1) and upside-down (|R[2,2]| ~ 1), and
    accepts near-90-degree tilts where the local z-axis lies near the XY plane.
    """

    def __init__(self, theta_deg: float = 88.0, require_floor_contact: bool = True):
        super().__init__()
        self.cos_thresh = np.cos(np.deg2rad(theta_deg))
        self.require_floor_contact = require_floor_contact

    def __call__(self, arg):
        # arg is an ObjectState or SiteObjectState; get quaternion
        quat = arg.get_geom_state()["quat"]  # (w, x, y, z)
        R = T.quat2mat(T.convert_quat(quat, to="xyzw"))  # columns are local axes in world
        z_world = abs(R[2, 2])  # world z component of local z-axis
        on_side = z_world < self.cos_thresh

        if not self.require_floor_contact:
            return bool(on_side)

        # Check contact with a fixture named "floor" if accessible
        try:
            floor_state = arg.env.object_states_dict.get("floor", None)
            if floor_state is None:
                # No explicit floor fixture tracked; fallback to orientation-only
                return bool(on_side)
            return bool(on_side and arg.check_contact(floor_state))
        except Exception:
            # Be permissive if env lookups fail
            return bool(on_side)


class OnSideWithRecentSupport(BinaryAtomic):
    """
    True iff:
    - The object is currently on its side (orientation-only), AND
    - The environment has latched that the wall was used as support during the
      most recent flip onto its side.

    Implementation details:
    - The environment (domain) tracks contact and OnSide histories per control
      step. Upon the first rising edge of OnSide (t_flip), it latches
      `support_granted=True` if contact(obj, wall) occurred within the configured
      pre-window before t_flip or occurs within a short post-window after t_flip.
      Once granted, it remains True for that flip. If the object later leaves
      the side pose, flip markers and latches are cleared so the next flip must
      again demonstrate support.

    Notes:
    - All windows are specified in control steps (env.step quanta), not MuJoCo
      integrator substeps, to align with success hold behavior.
    - If the environment does not provide the tracking attributes, we fall back
      to the instantaneous condition: OnSide(now) AND InContact(now).
    """

    def __init__(self, theta_deg: float = 88.0):
        super().__init__()
        self.cos_thresh = np.cos(np.deg2rad(theta_deg))

    def __call__(self, obj_state, wall_state):
        # Orientation-only OnSide check (match OnSide default)
        quat = obj_state.get_geom_state()["quat"]
        R = T.quat2mat(T.convert_quat(quat, to="xyzw"))
        z_world = abs(R[2, 2])
        on_side_now = z_world < self.cos_thresh

        env = getattr(obj_state, "env", None)
        if env is None:
            return False

        # Preferred: read latched support from env
        key = (obj_state.object_name, wall_state.object_name)
        try:
            support_latched = env._support_granted.get(key, False)
        except Exception:
            support_latched = None

        if support_latched is None:
            # Fallback instantaneous: require contact now
            try:
                return bool(on_side_now and obj_state.check_contact(wall_state))
            except Exception:
                return bool(on_side_now)

        return bool(on_side_now and support_latched)
