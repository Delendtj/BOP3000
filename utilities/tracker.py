def tracking(tracker, input, frame, org_shape):
    # Reshaper fra (1,300,6) til (300,6)
    data = input.reshape((300,6))

    # Resizer bilde til original bilde størrelse
    for det in data:
        det[0] = det[0]/640 * org_shape[1]
        det[1] = det[1]/640 * org_shape[0]
        det[2] = det[2]/640 * org_shape[1]
        det[3] = det[3]/640 * org_shape[0]

    # Tracker/legger faktisk på id til detections.
    tracker.update(data, frame)
    # Deretter gjør det synlig ved å plotte bbox/stats på framen direkte.
    tracker.plot_results(frame, fontscale=1, show_lost=True, show_trajectories=False)


