def heuristic_policy(obs, env):
    """PD controller adapted from gymnasium's reference LunarLander heuristic.

    Steers attitude toward an angle proportional to (x, vx) so the lander
    leans back toward the pad, and fires the main engine when the descent
    is faster than a hover target proportional to |x|. On ground contact
    only the descent is damped.

    Returns one of the 4 discrete actions:
        0 noop, 1 left engine, 2 main engine, 3 right engine.
    """
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = obs

    angle_target = max(-0.4, min(0.4, x * 0.5 + vx * 1.0))
    hover_target = 0.55 * abs(x)

    angle_todo = (angle_target - angle) * 0.5 - angular_vel * 1.0
    hover_todo = (hover_target - y) * 0.5 - vy * 0.5

    if left_contact or right_contact:
        angle_todo = 0
        hover_todo = -vy * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0
