# ==========================================================
# AERO60492 Coursework 3 - Feedback Control
#
# Overall control idea:
#   This controller is designed to stabilise a UAV at a desired
#   3D position and yaw angle using velocity commands.
#
#   The controller first computes the position and yaw errors from
#   the current state and target. Since the horizontal target error is
#   naturally expressed in the world frame, it is transformed into the
#   yaw-aligned control frame before generating x/y velocity commands.
#
#   A PI-D feedback structure is then used for x, y, z and yaw control:
#     - proportional action drives the UAV towards the target;
#     - integral action reduces steady-state error;
#     - derivative action is implemented through filtered velocity
#       estimates to improve damping and reduce overshoot.
#
#   To improve robustness, a DOB-inspired compensation term is also
#   included. It estimates a filtered correction term from the difference
#   between the full PI-D command and the nominal proportional command,
#   then uses this estimate to compensate persistent tracking bias caused
#   by wind, modelling mismatch, command delay and accumulated correction.
#
#   The controller also uses far / mid / near gain scheduling and command
#   saturation. Far mode allows faster motion towards the target, while
#   near mode limits the command magnitude to improve final stabilisation.
#
#   During each run, flight data is logged and later exported to CSV and
#   summary plots. These logs are used for tuning analysis, 
#   but they do not directly affect the control output.
# ==========================================================

import math
import csv
import atexit

def controller(state, target_pos, dt, wind_enabled=False):
    # ==========================================================
    # Controller type:
    #   Disturbance-Observer-Based PI-D Feedback Controller
    #   with yaw-frame coordinate transformation and gain scheduling.
    #
    # Coursework objective:
    #   The UAV must reach and stabilise at a user-defined 3D position
    #   and yaw angle. The controller receives the current UAV state and
    #   outputs velocity commands in x, y, z and yaw rate.
    #
    # Advanced method used:
    #   This controller uses a DOB-inspired compensation structure.
    #   Instead of using a full UAV dynamic model, it estimates a filtered
    #   correction term from the difference between the PI-D feedback command
    #   and the nominal proportional command.
    #   This term captures persistent extra control effort caused by wind,
    #   modelling mismatch, command delay, velocity damping and integral
    #   correction. It is then subtracted from the baseline PI-D command to
    #   improve robustness and reduce steady-state tracking error.
    #
    # Main control structure:
    #   1. Read current position and yaw.
    #   2. Compute position/yaw error.
    #   3. Transform horizontal position error from world frame into the
    #      yaw-aligned control frame.
    #   4. Estimate velocity using filtered finite differences.
    #   5. Apply PI-D feedback control.
    #   6. Update disturbance observer estimates.
    #   7. Apply DOB compensation.
    #   8. Apply command saturation for safe and stable behaviour.
    #
    # Experimental functions:
    #   This file also records flight data into CSV and saves a final
    #   figure. These outputs are used as experimental evidence.
    # ==========================================================

    # ---------------- Utility functions ----------------
    # Clamp limits the command or internal state to a safe range.
    # This prevents excessive control commands and also implements
    # anti-windup for the integral terms.
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    # Wrap any angle into [-pi, pi].
    # This is essential for yaw control because yaw is periodic.
    def wrap_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    # Sign helper used for the small minimum closing-speed logic.
    def sign(v):
        if v > 0.0:
            return 1.0
        if v < 0.0:
            return -1.0
        return 0.0

    # Persistent memory
    # The simulator repeatedly calls controller().
    # Function attributes are used to store states between calls without
    # changing the required controller input/output interface.
    if not hasattr(controller, "init_done"):
        controller.init_done = True

        # Previous state values used for velocity estimation.
        # Velocity is estimated using finite differences and then filtered.
        controller.prev_x = 0.0
        controller.prev_y = 0.0
        controller.prev_z = 0.0
        controller.prev_yaw = 0.0

        # Filtered velocity estimates.
        # These are used as derivative feedback terms.
        # The derivative action improves damping and reduces overshoot.
        controller.vx_hat_w = 0.0
        controller.vy_hat_w = 0.0
        controller.vz_hat = 0.0
        controller.vyaw_hat = 0.0

        # Integral states.
        # These help remove steady-state error, especially under constant
        # disturbances such as wind or small command bias.
        controller.int_ex = 0.0
        controller.int_ey = 0.0
        controller.int_ez = 0.0
        controller.int_eyaw = 0.0

        # DOB states.
        # These represent estimated disturbance inx, y, z and yaw channels.
        controller.dx_hat = 0.0
        controller.dy_hat = 0.0
        controller.dz_hat = 0.0
        controller.dyaw_hat = 0.0

        # Previous target / position for reset logic.
        # These are used to detect target changes or simulator resets.
        controller.prev_target = None
        controller.prev_pos = None

        # Previous error states.
        controller.prev_ex = 0.0
        controller.prev_ey = 0.0
        controller.prev_ez = 0.0
        controller.prev_eyaw = 0.0

        # Simulation time for logging.
        controller.sim_time = 0.0

        # Realtime plotting disabled for performance.
        # The code only saves the final figure at the end.
        controller.realtime_plot_enabled = False

        # Final output files used for experiment evidence.
        controller.final_csv_filename = "controller_logs.csv"
        controller.final_figure_filename = "controller_realtime_figure.png"

        # Prevent repeated save.
        controller.exit_save_done = False

        # Logs.
        # These are used to generate quantitative evidence for the video:
        # position tracking, error convergence, control commands and DOB
        # disturbance estimates.
        controller.t_log = []

        controller.x_log = []
        controller.y_log = []
        controller.z_log = []
        controller.yaw_log = []

        controller.tx_log = []
        controller.ty_log = []
        controller.tz_log = []
        controller.tyaw_log = []

        controller.ex_log = []
        controller.ey_log = []
        controller.ez_log = []
        controller.eyaw_log = []

        controller.vx_cmd_log = []
        controller.vy_cmd_log = []
        controller.vz_cmd_log = []
        controller.yaw_rate_cmd_log = []

        controller.vx_hat_log = []
        controller.vy_hat_log = []
        controller.vz_hat_log = []
        controller.vyaw_hat_log = []

        controller.dx_hat_log = []
        controller.dy_hat_log = []
        controller.dz_hat_log = []
        controller.dyaw_hat_log = []

        controller.mode_log = []

    # Safety fallback.
    # If the simulator gives an invalid dt, use a small default time step.
    # This prevents division-by-zero in the velocity estimator.
    if dt is None or dt <= 1e-6:
        dt = 0.01

    controller.sim_time += dt

    # Unpack current state and target
    # State is given as current UAV position and attitude.
    # Roll and pitch are available but this controller mainly uses yaw,
    # because the command frame is yaw-aligned in this simulator setup.
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # Reset logic
    reset_needed = False

    # Detect large target change.
    if controller.prev_target is not None:
        if (
            abs(tx - controller.prev_target[0]) > 0.5 or
            abs(ty - controller.prev_target[1]) > 0.5 or
            abs(tz - controller.prev_target[2]) > 0.5 or
            abs(wrap_angle(tyaw - controller.prev_target[3])) > 0.5
        ):
            reset_needed = True

    # Detect simulator reset or large jump in measured position.
    if controller.prev_pos is not None:
        jump = math.sqrt(
            (x - controller.prev_pos[0]) ** 2 +
            (y - controller.prev_pos[1]) ** 2 +
            (z - controller.prev_pos[2]) ** 2
        )
        if jump > 1.2:
            reset_needed = True

    # Reset velocity estimates, integrators and disturbance estimates.
    # This avoids carrying old accumulated error into a new target test.
    if reset_needed:
        controller.vx_hat_w = 0.0
        controller.vy_hat_w = 0.0
        controller.vz_hat = 0.0
        controller.vyaw_hat = 0.0

        controller.int_ex = 0.0
        controller.int_ey = 0.0
        controller.int_ez = 0.0
        controller.int_eyaw = 0.0

        controller.dx_hat = 0.0
        controller.dy_hat = 0.0
        controller.dz_hat = 0.0
        controller.dyaw_hat = 0.0

    # Errors in world frame
    # Position error is first calculated in the world/global frame.
    # yaw error is wrapped so that the controller always turns through
    # the shortest angular direction.
    ex_w = tx - x
    ey_w = ty - y
    ez = tz - z
    eyaw = wrap_angle(tyaw - yaw)

    # Velocity estimation
    # The controller does not assume direct velocity measurements.
    # Instead, it estimates velocity from consecutive position samples.
    # A first-order low-pass filter is then applied to reduce noise.
    # These filtered velocities are used as derivative feedback terms.
    # This makes the position response less oscillatory.
    raw_vx_w = (x - controller.prev_x) / dt
    raw_vy_w = (y - controller.prev_y) / dt
    raw_vz = (z - controller.prev_z) / dt
    raw_vyaw = wrap_angle(yaw - controller.prev_yaw) / dt

    vel_alpha = 0.25
    controller.vx_hat_w = (1.0 - vel_alpha) * controller.vx_hat_w + vel_alpha * raw_vx_w
    controller.vy_hat_w = (1.0 - vel_alpha) * controller.vy_hat_w + vel_alpha * raw_vy_w
    controller.vz_hat = (1.0 - vel_alpha) * controller.vz_hat + vel_alpha * raw_vz
    controller.vyaw_hat = (1.0 - vel_alpha) * controller.vyaw_hat + vel_alpha * raw_vyaw


    # Transform to yaw-aligned control frame
    # The target position error is naturally computed in the world frame,
    # but the UAV velocity commands are interpreted relative to the UAV
    # yaw-aligned control frame. Therefore, the horizontal error and
    # horizontal velocity estimate are rotated by the current yaw angle.
    # This allows vx_cmd to mean "move forward/backward relative to the
    # current yaw" and vy_cmd to mean "move sideways relative to the
    # current yaw".
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    ex = cy * ex_w + sy * ey_w
    ey = -sy * ex_w + cy * ey_w

    vx_b = cy * controller.vx_hat_w + sy * controller.vy_hat_w
    vy_b = -sy * controller.vx_hat_w + cy * controller.vy_hat_w

    pos_err_xy = math.sqrt(ex_w * ex_w + ey_w * ey_w)
    pos_err = math.sqrt(ex_w * ex_w + ey_w * ey_w + ez * ez)

    # Gains
    # Two sets of gains are used:
    #   - no-wind mode: tuned for nominal simulator behaviour
    #   - wind mode: slightly stronger feedback and DOB gains for
    #     improved disturbance rejection
    #
    # These parameters were tuned experimentally by observing response
    # speed, overshoot, steady-state error and stability.
    if wind_enabled:
        kp_xy = 1.05
        ki_xy = 0.10
        kd_xy = 0.32

        kp_z = 1.20
        ki_z = 0.12
        kd_z = 0.28

        kp_yaw = 1.90
        ki_yaw = 0.05
        kd_yaw = 0.16

        dob_gain_xy = 0.18
        dob_gain_z = 0.20
        dob_gain_yaw = 0.14

        vxy_max_far = 0.90
        vxy_max_mid = 0.40
        vxy_max_near = 0.16

        vz_max_far = 0.70
        vz_max_mid = 0.28
        vz_max_near = 0.12

        yaw_rate_max_far = 0.95
        yaw_rate_max_mid = 0.32
        yaw_rate_max_near = 0.16
    else:
        kp_xy = 1.00
        ki_xy = 0.08
        kd_xy = 0.28

        kp_z = 1.14
        ki_z = 0.10
        kd_z = 0.24

        kp_yaw = 1.82
        ki_yaw = 0.04
        kd_yaw = 0.14

        dob_gain_xy = 0.14
        dob_gain_z = 0.16
        dob_gain_yaw = 0.11

        vxy_max_far = 0.82
        vxy_max_mid = 0.36
        vxy_max_near = 0.14

        vz_max_far = 0.62
        vz_max_mid = 0.25
        vz_max_near = 0.10

        yaw_rate_max_far = 0.84
        yaw_rate_max_mid = 0.30
        yaw_rate_max_near = 0.14

    # 3 modes: far / mid / near
    # Gain scheduling is used to improve behaviour across different
    # distance ranges.
    #
    # Far mode:
    #   Allows larger commands so that the UAV moves quickly toward the
    #   target.
    #
    # Mid mode:
    #   Slightly increases damping and integral action to reduce overshoot
    #   while still maintaining convergence speed.
    #
    # Near mode:
    #   Reduces command limits and increases fine correction capability.
    #   This helps achieve a low final mean error and low standard deviation.
    far_mode = pos_err > 0.35
    mid_mode = 0.12 < pos_err <= 0.35
    near_mode = pos_err <= 0.12

    if far_mode:
        kp_xy_use = kp_xy
        ki_xy_use = ki_xy
        kd_xy_use = kd_xy

        kp_z_use = kp_z
        ki_z_use = ki_z
        kd_z_use = kd_z

        kp_yaw_use = kp_yaw
        ki_yaw_use = ki_yaw
        kd_yaw_use = kd_yaw

        vxy_max = vxy_max_far
        vz_max = vz_max_far
        yaw_rate_max = yaw_rate_max_far
        mode_name = "far"

    elif mid_mode:
        kp_xy_use = kp_xy * 0.98
        ki_xy_use = ki_xy * 1.10
        kd_xy_use = kd_xy * 1.05

        kp_z_use = kp_z * 0.98
        ki_z_use = ki_z * 1.10
        kd_z_use = kd_z * 1.05

        kp_yaw_use = kp_yaw * 0.98
        ki_yaw_use = ki_yaw * 1.05
        kd_yaw_use = kd_yaw * 1.03

        vxy_max = vxy_max_mid
        vz_max = vz_max_mid
        yaw_rate_max = yaw_rate_max_mid
        mode_name = "mid"

    else:
        kp_xy_use = kp_xy * 1.04
        ki_xy_use = ki_xy * 1.35
        kd_xy_use = kd_xy * 1.18

        kp_z_use = kp_z * 1.00
        ki_z_use = ki_z * 1.05
        kd_z_use = kd_z * 1.08

        kp_yaw_use = kp_yaw * 0.98
        ki_yaw_use = ki_yaw * 1.05
        kd_yaw_use = kd_yaw * 1.08

        vxy_max = vxy_max_near
        vz_max = vz_max_near
        yaw_rate_max = yaw_rate_max_near
        mode_name = "near"

    # Integrators
    # Integral terms reduce steady-state error caused by constant
    # disturbances, small model mismatch, or command bias.
    # The integral terms are only actively accumulated when the error is
    # above a small threshold. Near zero error, the integrator slowly leaks
    # away. This avoids integral windup and reduces final oscillation.
    allow_xy_int = pos_err_xy > 0.006
    allow_z_int = abs(ez) > 0.008
    allow_yaw_int = abs(eyaw) > 0.015

    if allow_xy_int:
        controller.int_ex += ex * dt
        controller.int_ey += ey * dt
    else:
        controller.int_ex *= 0.998
        controller.int_ey *= 0.998

    if allow_z_int:
        controller.int_ez += ez * dt
    else:
        controller.int_ez *= 0.997

    if allow_yaw_int:
        controller.int_eyaw += eyaw * dt
    else:
        controller.int_eyaw *= 0.997

    # Anti-windup limits.
    # These prevent the integral terms from becoming too large.
    controller.int_ex = clamp(controller.int_ex, -0.90, 0.90)
    controller.int_ey = clamp(controller.int_ey, -0.90, 0.90)
    controller.int_ez = clamp(controller.int_ez, -0.70, 0.70)
    controller.int_eyaw = clamp(controller.int_eyaw, -0.35, 0.35)

    # Baseline PI-D feedback
    # PI-D structure:
    #   P term:
    #       Drives the UAV toward the target.
    #
    #   I term:
    #       Removes persistent steady-state error.
    #
    #   D term:
    #       Uses estimated velocity as damping.
    #
    # This is implemented as velocity command feedback rather than direct
    # force or attitude control
    vx_fb = kp_xy_use * ex + ki_xy_use * controller.int_ex - kd_xy_use * vx_b
    vy_fb = kp_xy_use * ey + ki_xy_use * controller.int_ey - kd_xy_use * vy_b
    vz_fb = kp_z_use * ez + ki_z_use * controller.int_ez - kd_z_use * controller.vz_hat
    yaw_fb = kp_yaw_use * eyaw + ki_yaw_use * controller.int_eyaw - kd_yaw_use * controller.vyaw_hat

    # -------------------------------------------------
    # Disturbance observer update
    # -------------------------------------------------
    # The DOB-like estimator tracks the difference between the full PI-D
    # feedback command and the nominal proportional command.
    #
    # Interpretation:
    #   If a persistent correction is repeatedly required beyond the
    #   proportional term, the observer treats it as an estimated
    #   disturbance or modelling mismatch.
    # The observer is filtered by dob_gain_* so that the disturbance
    # estimate changes smoothly rather than reacting aggressively to noise.
    controller.dx_hat = (1.0 - dob_gain_xy) * controller.dx_hat + dob_gain_xy * (vx_fb - kp_xy_use * ex)
    controller.dy_hat = (1.0 - dob_gain_xy) * controller.dy_hat + dob_gain_xy * (vy_fb - kp_xy_use * ey)
    controller.dz_hat = (1.0 - dob_gain_z) * controller.dz_hat + dob_gain_z * (vz_fb - kp_z_use * ez)
    controller.dyaw_hat = (1.0 - dob_gain_yaw) * controller.dyaw_hat + dob_gain_yaw * (yaw_fb - kp_yaw_use * eyaw)

    # Limit the estimated disturbance.
    # This prevents the DOB term from dominating the feedback controller.
    controller.dx_hat = clamp(controller.dx_hat, -0.18, 0.18)
    controller.dy_hat = clamp(controller.dy_hat, -0.18, 0.18)
    controller.dz_hat = clamp(controller.dz_hat, -0.14, 0.14)
    controller.dyaw_hat = clamp(controller.dyaw_hat, -0.10, 0.10)

    # DOB compensation
    # The estimated disturbance is subtracted from the baseline feedback
    # command. This compensates for persistent bias and improves robustness,
    # especially when wind_enabled is True.
    vx_cmd = vx_fb - controller.dx_hat
    vy_cmd = vy_fb - controller.dy_hat
    vz_cmd = vz_fb - controller.dz_hat
    yaw_rate_cmd = yaw_fb - controller.dyaw_hat

    # Gentle minimum effective closing speed in XY only
    # Some simulated or real devices may not respond well to extremely
    # small velocity commands. In near mode, this block ensures that if
    # there is still a small but meaningful horizontal error, the command
    # remains large enough to keep closing the error.
    # This is only applied in XY and only near the target.
    if near_mode:
        if abs(ex) > 0.002:
            min_vx = min(0.040, 0.16 * abs(ex) + 0.010)
            if abs(vx_cmd) < min_vx:
                vx_cmd = sign(ex) * min_vx

        if abs(ey) > 0.002:
            min_vy = min(0.040, 0.16 * abs(ey) + 0.010)
            if abs(vy_cmd) < min_vy:
                vy_cmd = sign(ey) * min_vy

    # Saturation
    # Command saturation keeps the velocity commands inside safe and
    # stable bounds. The bounds depend on whether the UAV is far from,
    # near to, or very close to the target.
    vx_cmd = clamp(vx_cmd, -vxy_max, vxy_max)
    vy_cmd = clamp(vy_cmd, -vxy_max, vxy_max)
    vz_cmd = clamp(vz_cmd, -vz_max, vz_max)
    yaw_rate_cmd = clamp(yaw_rate_cmd, -yaw_rate_max, yaw_rate_max)

    # Logging
    # Store all useful control data for post-flight analysis.
    # These logs are used to generate:
    #   - tracking plots
    #   - error plots
    #   - command plots
    #   - DOB estimate plots

    controller.t_log.append(controller.sim_time)

    controller.x_log.append(x)
    controller.y_log.append(y)
    controller.z_log.append(z)
    controller.yaw_log.append(yaw)

    controller.tx_log.append(tx)
    controller.ty_log.append(ty)
    controller.tz_log.append(tz)
    controller.tyaw_log.append(tyaw)

    controller.ex_log.append(ex)
    controller.ey_log.append(ey)
    controller.ez_log.append(ez)
    controller.eyaw_log.append(eyaw)

    controller.vx_cmd_log.append(vx_cmd)
    controller.vy_cmd_log.append(vy_cmd)
    controller.vz_cmd_log.append(vz_cmd)
    controller.yaw_rate_cmd_log.append(yaw_rate_cmd)

    controller.vx_hat_log.append(vx_b)
    controller.vy_hat_log.append(vy_b)
    controller.vz_hat_log.append(controller.vz_hat)
    controller.vyaw_hat_log.append(controller.vyaw_hat)

    controller.dx_hat_log.append(controller.dx_hat)
    controller.dy_hat_log.append(controller.dy_hat)
    controller.dz_hat_log.append(controller.dz_hat)
    controller.dyaw_hat_log.append(controller.dyaw_hat)

    controller.mode_log.append(mode_name)

    # Save previous states
    # These values are needed at the next controller call for:
    #   - velocity estimation
    #   - reset detection
    #   - target-change detection
    controller.prev_x = x
    controller.prev_y = y
    controller.prev_z = z
    controller.prev_yaw = yaw

    controller.prev_ex = ex
    controller.prev_ey = ey
    controller.prev_ez = ez
    controller.prev_eyaw = eyaw

    controller.prev_target = (tx, ty, tz, tyaw)
    controller.prev_pos = (x, y, z)

    return (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)


def reset_controller_logs():
    """
    Clear all stored logs and reset the internal simulation clock.
    This function is useful when running multiple experiments manually.
    It does not affect the controller design itself. It only clears the
    data used for CSV and plotting.
    """
    if hasattr(controller, "t_log"):
        controller.t_log.clear()

        controller.x_log.clear()
        controller.y_log.clear()
        controller.z_log.clear()
        controller.yaw_log.clear()

        controller.tx_log.clear()
        controller.ty_log.clear()
        controller.tz_log.clear()
        controller.tyaw_log.clear()

        controller.ex_log.clear()
        controller.ey_log.clear()
        controller.ez_log.clear()
        controller.eyaw_log.clear()

        controller.vx_cmd_log.clear()
        controller.vy_cmd_log.clear()
        controller.vz_cmd_log.clear()
        controller.yaw_rate_cmd_log.clear()

        controller.vx_hat_log.clear()
        controller.vy_hat_log.clear()
        controller.vz_hat_log.clear()
        controller.vyaw_hat_log.clear()

        controller.dx_hat_log.clear()
        controller.dy_hat_log.clear()
        controller.dz_hat_log.clear()
        controller.dyaw_hat_log.clear()

        controller.mode_log.clear()

        controller.sim_time = 0.0
        controller.exit_save_done = False


def save_controller_logs_csv(filename="controller_logs.csv"):
    """
    Save all logged controller data to a CSV file.
      - tracking accuracy
      - steady-state error
      - command saturation
      - DOB compensation behaviour
      - tuning comparison between different controller versions
    """
    if not hasattr(controller, "t_log") or len(controller.t_log) == 0:
        print("No log data available. CSV was not saved.")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "x", "y", "z", "yaw",
            "tx", "ty", "tz", "tyaw",
            "ex", "ey", "ez", "eyaw",
            "vx_cmd", "vy_cmd", "vz_cmd", "yaw_rate_cmd",
            "vx_hat_b", "vy_hat_b", "vz_hat", "vyaw_hat",
            "dx_hat", "dy_hat", "dz_hat", "dyaw_hat",
            "mode"
        ])

        for i in range(len(controller.t_log)):
            writer.writerow([
                controller.t_log[i],
                controller.x_log[i],
                controller.y_log[i],
                controller.z_log[i],
                controller.yaw_log[i],
                controller.tx_log[i],
                controller.ty_log[i],
                controller.tz_log[i],
                controller.tyaw_log[i],
                controller.ex_log[i],
                controller.ey_log[i],
                controller.ez_log[i],
                controller.eyaw_log[i],
                controller.vx_cmd_log[i],
                controller.vy_cmd_log[i],
                controller.vz_cmd_log[i],
                controller.yaw_rate_cmd_log[i],
                controller.vx_hat_log[i],
                controller.vy_hat_log[i],
                controller.vz_hat_log[i],
                controller.vyaw_hat_log[i],
                controller.dx_hat_log[i],
                controller.dy_hat_log[i],
                controller.dz_hat_log[i],
                controller.dyaw_hat_log[i],
                controller.mode_log[i]
            ])

    print(f"Saved CSV to {filename}")


def save_controller_final_figure(filename="controller_realtime_figure.png"):
    """
    Generate and save the final 2x2 summary figure from stored logs.
    The generated figure summarises:
      1. Position tracking
      2. Tracking errors
      3. Control commands
      4. DOB disturbance estimates
    """
    if not hasattr(controller, "t_log") or len(controller.t_log) == 0:
        print("No log data available. Figure was not saved.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        t = controller.t_log

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Final UAV Flight Data")

        ax_pos = axes[0, 0]
        ax_err = axes[0, 1]
        ax_cmd = axes[1, 0]
        ax_dob = axes[1, 1]

        # Position plot.
        # This shows whether the measured x, y, z positions track the
        # commanded target positions.
        ax_pos.plot(t, controller.x_log, label="x")
        ax_pos.plot(t, controller.tx_log, "--", label="x_target")
        ax_pos.plot(t, controller.y_log, label="y")
        ax_pos.plot(t, controller.ty_log, "--", label="y_target")
        ax_pos.plot(t, controller.z_log, label="z")
        ax_pos.plot(t, controller.tz_log, "--", label="z_target")
        ax_pos.set_title("Position Tracking")
        ax_pos.set_xlabel("Time [s]")
        ax_pos.set_ylabel("Position")
        ax_pos.grid(True)
        ax_pos.legend(loc="best", fontsize=8)

        # Error plot.
        # This shows convergence speed, overshoot and final steady-state
        # accuracy for x, y, z and yaw.
        ax_err.plot(t, controller.ex_log, label="ex")
        ax_err.plot(t, controller.ey_log, label="ey")
        ax_err.plot(t, controller.ez_log, label="ez")
        ax_err.plot(t, controller.eyaw_log, label="eyaw")
        ax_err.set_title("Tracking Errors")
        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error")
        ax_err.grid(True)
        ax_err.legend(loc="best", fontsize=8)

        # Command plot.
        # This shows the output sent by the controller. It helps explain
        # whether the controller was smooth, aggressive or saturated.
        ax_cmd.plot(t, controller.vx_cmd_log, label="vx_cmd")
        ax_cmd.plot(t, controller.vy_cmd_log, label="vy_cmd")
        ax_cmd.plot(t, controller.vz_cmd_log, label="vz_cmd")
        ax_cmd.plot(t, controller.yaw_rate_cmd_log, label="yaw_rate_cmd")
        ax_cmd.set_title("Control Commands")
        ax_cmd.set_xlabel("Time [s]")
        ax_cmd.set_ylabel("Command")
        ax_cmd.grid(True)
        ax_cmd.legend(loc="best", fontsize=8)

        # DOB plot.
        # This shows the estimated disturbance terms used by the advanced
        # controller. It provides evidence that the DOB-like compensation
        # is active during the flight.
        ax_dob.plot(t, controller.dx_hat_log, label="dx_hat")
        ax_dob.plot(t, controller.dy_hat_log, label="dy_hat")
        ax_dob.plot(t, controller.dz_hat_log, label="dz_hat")
        ax_dob.plot(t, controller.dyaw_hat_log, label="dyaw_hat")
        ax_dob.set_title("DOB Estimates")
        ax_dob.set_xlabel("Time [s]")
        ax_dob.set_ylabel("Estimated Disturbance")
        ax_dob.grid(True)
        ax_dob.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved final figure to {filename}")

    except Exception as e:
        print(f"Final figure save failed: {e}")


def save_controller_outputs():
    """
    Save final CSV and final figure once.
    """
    if not hasattr(controller, "init_done"):
        return

    if not hasattr(controller, "t_log") or len(controller.t_log) == 0:
        print("No log data available. Outputs were not saved.")
        return

    save_controller_logs_csv(controller.final_csv_filename)
    save_controller_final_figure(controller.final_figure_filename)


def auto_save_on_exit():
    """
    Automatically save final CSV and final figure when the program exits normally.
    This is included so that experimental data is not lost when the simulator
    is closed using the normal quit command.
    """
    if not hasattr(controller, "init_done"):
        return

    if getattr(controller, "exit_save_done", False):
        return

    controller.exit_save_done = True

    try:
        save_controller_outputs()
    except Exception as e:
        print(f"Auto-save failed: {e}")


# Register automatic saving at program exit.
atexit.register(auto_save_on_exit)


def plot_controller_logs():
    """
    It is useful during tuning because it allows visual inspection of:
      - position tracking
      - yaw tracking
      - tracking errors
      - control commands
      - estimated velocities
      - DOB estimates
    """
    if not hasattr(controller, "t_log") or len(controller.t_log) == 0:
        print("No log data available.")
        return

    import matplotlib.pyplot as plt

    t = controller.t_log

    # Position tracking plot.
    plt.figure(figsize=(10, 6))
    plt.plot(t, controller.x_log, label="x")
    plt.plot(t, controller.tx_log, "--", label="x_target")
    plt.plot(t, controller.y_log, label="y")
    plt.plot(t, controller.ty_log, "--", label="y_target")
    plt.plot(t, controller.z_log, label="z")
    plt.plot(t, controller.tz_log, "--", label="z_target")
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.title("Position Tracking")
    plt.legend()
    plt.grid(True)

    # Yaw tracking plot.
    plt.figure(figsize=(10, 5))
    plt.plot(t, controller.yaw_log, label="yaw")
    plt.plot(t, controller.tyaw_log, "--", label="yaw_target")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")
    plt.title("Yaw Tracking")
    plt.legend()
    plt.grid(True)

    # Error convergence plot.
    plt.figure(figsize=(10, 6))
    plt.plot(t, controller.ex_log, label="ex")
    plt.plot(t, controller.ey_log, label="ey")
    plt.plot(t, controller.ez_log, label="ez")
    plt.plot(t, controller.eyaw_log, label="eyaw")
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title("Tracking Errors")
    plt.legend()
    plt.grid(True)

    # Control command plot.
    plt.figure(figsize=(10, 6))
    plt.plot(t, controller.vx_cmd_log, label="vx_cmd")
    plt.plot(t, controller.vy_cmd_log, label="vy_cmd")
    plt.plot(t, controller.vz_cmd_log, label="vz_cmd")
    plt.plot(t, controller.yaw_rate_cmd_log, label="yaw_rate_cmd")
    plt.xlabel("Time [s]")
    plt.ylabel("Command")
    plt.title("Control Commands")
    plt.legend()
    plt.grid(True)

    # Estimated velocity plot.
    plt.figure(figsize=(10, 6))
    plt.plot(t, controller.vx_hat_log, label="vx_hat_b")
    plt.plot(t, controller.vy_hat_log, label="vy_hat_b")
    plt.plot(t, controller.vz_hat_log, label="vz_hat")
    plt.plot(t, controller.vyaw_hat_log, label="vyaw_hat")
    plt.xlabel("Time [s]")
    plt.ylabel("Estimated Velocity")
    plt.title("Estimated Velocities")
    plt.legend()
    plt.grid(True)

    # DOB estimate plot.
    plt.figure(figsize=(10, 6))
    plt.plot(t, controller.dx_hat_log, label="dx_hat")
    plt.plot(t, controller.dy_hat_log, label="dy_hat")
    plt.plot(t, controller.dz_hat_log, label="dz_hat")
    plt.plot(t, controller.dyaw_hat_log, label="dyaw_hat")
    plt.xlabel("Time [s]")
    plt.ylabel("Estimated Disturbance")
    plt.title("DOB Estimates")
    plt.legend()
    plt.grid(True)

    plt.show()