def heuristic_policy(obs, env):
    x, y, vx, vy, angle, angular_vel, left_contact, right_contact = obs

    if left_contact and right_contact:
        return  0
    if abs(angle) > 0.2:
        return 1 if angle > 0 else 3
    
    if vy < -0.5 or y > 0.5:
        return 2
    
    if x > 0.2:
        return 3
    if x > -0.2:
        return 1
    
    return 0