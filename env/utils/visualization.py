import cv2
import numpy as np


def visualize_RL_image(env, viz_env_id, infos, valid_mask, rewards, actions, observations=None, add_actions=True):
    feature_image = env.visualize_features(viz_env_id)
    if feature_image.sum() == 0:
        feature_image = infos[viz_env_id]['image'].copy()
        if feature_image.shape[2] == 1:
            feature_image = np.tile(feature_image, [1, 1, 3])

    # Add border to visualize valid state or not
    border = 3
    if not valid_mask[viz_env_id]:
        border_c = np.array([0, 0, 255])
    else:
        border_c = np.array([0, 255, 0])
    feature_image[:border, :, :] = border_c
    feature_image[-border:, :, :] = border_c
    feature_image[:, :border, :] = border_c
    feature_image[:, -border:, :] = border_c

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)  # Text color (white)
    line_type = 2
    top_text = "{: <14}   Reward: {:.4f}".format(
        env.get_stage(np.array([infos[viz_env_id]['vo_stages']]))[0],
        rewards[viz_env_id])
    (text_width, text_height), _ = cv2.getTextSize(top_text, font, font_scale, line_type)

    feature_image[-int(border+1.6*text_height):-border, border:-border, :] = 0
    feature_image[border:int(border+1.6*text_height), border:-border, :] = 0
    cv2.putText(feature_image, top_text, (5, text_height + border), font, font_scale, font_color, line_type)

    if add_actions:
        if actions.shape[1] == 2:
            bottom_text = "KF: {}    GridSize: {}".format(actions[viz_env_id, 0], actions[viz_env_id, 1])
        else:
            bottom_text = "KF: {}".format(actions[viz_env_id, 0])
        cv2.putText(feature_image, bottom_text, (5, feature_image.shape[0] - text_height // 2), font, font_scale,
                    font_color, line_type)

    keyframe_dist = np.round(observations[viz_env_id, 1])
    assert np.abs(observations[viz_env_id, 1] - int(keyframe_dist)) < 1e-5
    bottom_right_text = "Last KF: {}".format(int(observations[viz_env_id, 1]))
    cv2.putText(feature_image, bottom_right_text, (feature_image.shape[1] - 150,
                                                   feature_image.shape[0] - text_height // 2), font, font_scale,
                font_color, line_type)

    return feature_image


def add_text_to_image(image, text, position='topright'):
    if position is not 'topright':
        raise NotImplementedError

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (255, 255, 255)  # Text color (white)
    line_type = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_type)
    cv2.putText(image, text, (image.shape[1] - text_width - 5, text_height + 2), font, font_scale, font_color, line_type)

    return image
