def tracking(tracker, input, frame, org_shape, conf_threshold):
    # Reshaper fra (1,300,6) til (300,6)
    data = input.reshape((300, 6))

    # filtrer ut detections under conf fra cfg
    valid_mask = data[:, 4] > conf_threshold
    data = data[valid_mask]

    # ingen detections over conf_treshold
    if len(data) == 0:
        return None

    # Resizer bilde til original bilde størrelse
    for det in data:
        det[0] = det[0] / 640 * org_shape[1]
        det[1] = det[1] / 640 * org_shape[0]
        det[2] = det[2] / 640 * org_shape[1]
        det[3] = det[3] / 640 * org_shape[0]

    # Tracker/legger faktisk på id til detections.
    tracker.update(data, frame)
    tracker.plot_results(frame, fontscale=1, show_lost=True, show_trajectories=False)

    return data