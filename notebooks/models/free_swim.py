import numpy as np

def compute_basic_features(array, threshold, ligate, dt_minimum, framesBeforeDetection, framesAfterDetection, Fs):
    import copy
    # Summarize fish behavior by computing 1) number of events, 2) start indices, 3) end indices, 4) time between the initialization of each event (III), /
    # 5) duration of events, and 6) time between the end of one event and the beginning of the next
    # Find bouts separated by small amount of time
    bin_array = copy.deepcopy(array)
    bin_array[array<threshold] = 0 
    bin_array[array>=threshold] = 1
    run_values, run_starts, run_lengths = find_runs(bin_array)
    # Modify start and end of swim bouts to account for rolling std window
    run_starts = run_starts[run_values == 1]
    run_lengths = run_lengths[run_values == 1]
    numSwims = len(run_starts)
    for k in range(numSwims):
        low = run_starts[k]
        high = run_starts[k] + run_lengths[k]
        # Modify
        if low >= framesBeforeDetection:
            low += framesBeforeDetection
        if high < len(array) - framesAfterDetection:
            high += framesAfterDetection
        bin_array[low:high] = 1
    # Swim detection again
    run_values, run_starts, run_lengths = find_runs(bin_array)
    # Skip if no bouts
    if np.sum(bin_array) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    if ligate:
        camera_time = np.arange(len(tailAngle)) / Fs
        run_lengths_t = convert_run_lengths(run_lengths, run_starts, camera_time)
        bin_array = fill_short_periods(bin_array, run_values, run_starts, run_lengths, run_lengths_t, dt_minimum) # fill in short no-motion periods (less than dt_minimum)
        run_values, run_starts, run_lengths = find_runs(bin_array) # recompute
    # Ignore motion that began before this period
    if run_values[0] == 1:
        run_values = run_values[1:]
        run_starts = run_starts[1:]
        run_lengths = run_lengths[1:]
    # Skip if no bouts remain
    if len(run_values) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    # Ignore motion that continues into the next period
    if run_values[-1] == 1:
        run_values = run_values[:-1]
        run_starts = run_starts[:-1]
        run_lengths = run_lengths[:-1]
    # Skip if no bouts remain
    if len(run_values) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    # Behavior features computed
    numEvents = np.sum(run_values)
    eventStarts = run_starts[run_values == 1] # indices of bout starts
    eventEnds = eventStarts + run_lengths[run_values == 1] # indices of bout ends
    eventIII = np.diff(run_starts[run_values == 1]) / Fs # inter-initialization intervals in seconds
    eventDur = run_lengths[run_values == 1] / Fs # swim durations in seconds
    # Ignore non-motion-periods that began before this period
    if run_values[0] == 0:
        run_values = run_values[1:]
        run_starts = run_starts[1:]
        run_lengths = run_lengths[1:]
    # Ignore non-motion-periods that continue into the next period
    if run_values[-1] == 0:
        run_values = run_values[:-1]
        run_starts = run_starts[:-1]
        run_lengths = run_lengths[:-1]
    # Bout features computed
    eventIBI = run_lengths[run_values == 0] / Fs # interbout intervals in seconds
    return numEvents, eventStarts, eventEnds, eventIII, eventDur, eventIBI

def find_runs(x_func):
    """Find runs of consecutive items in an array."""
    # ensure array
    x_func = np.asanyarray(x_func)
    if x_func.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x_func.shape[0]
    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x_func[:-1], x_func[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        # find run values
        run_values = x_func[loc_run_start]
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        return run_values, run_starts, run_lengths

def convert_run_lengths(run_lengths, run_starts, ct):
    # Convert frames to camera_time (i.e., seconds)
    run_lengths_t = np.zeros((len(run_lengths),))
    for i in range(len(run_lengths)):
        low = run_starts[i]
        high = run_starts[i] + run_lengths[i] - 1
        run_lengths_t[i] = ct[high] - ct[low]
    return run_lengths_t

def fill_short_periods(bin_tv, run_values, run_starts, run_lengths, run_lengths_t, dt_minimum):
    # Fill short lulls of low tail_vigor between swims if length of lulls â‰¤ dt_minimum
    inds = np.where(np.logical_and(run_values == 0, run_lengths_t < dt_minimum))[0]
    # Remove anything before first swim or after last swim
    inds = np.setdiff1d(inds, [0, len(run_values)])
    for i in inds:
        low = run_starts[i]
        high = run_starts[i] + run_lengths[i]
        bin_tv[low:high] = 1
    return bin_tv

def compute_event_intersection(event_a_starts, event_a_ends, event_b_starts, even_b_ends, numFrames, framesBeforeDetection, framesAfterDetection, Fs):
    # Compute the intersection of events
    a = np.zeros((numFrames,))
    b = np.zeros((numFrames,))
    # Make binary vectors of both events
    for i in range(len(event_a_starts)):
        a[event_a_starts[i]:event_a_ends[i]] = 1
    for i in range(len(event_b_starts)):
        b[event_b_starts[i]:even_b_ends[i]] = 1
    # Intersection
    commonEvents = np.logical_and(a, b)
    # Process
    numEvents, eventStarts, eventEnds, eventIII, eventDur, eventIBI = compute_basic_features(commonEvents, 0.5, 0, np.inf, framesBeforeDetection, framesAfterDetection, Fs)
    return commonEvents, numEvents, eventStarts, eventEnds, eventIII, eventDur, eventIBI

def compute_event_nonIntersection(event_a_starts, event_a_ends, event_b_starts, even_b_ends, numFrames, framesBeforeDetection, framesAfterDetection, Fs):
    # Compute the intersection of eventA not happening and eventB happening
    a = np.ones((numFrames,))
    b = np.zeros((numFrames,))
    # Make binary vectors of both events
    for i in range(len(event_a_starts)):
        a[event_a_starts[i]:event_a_ends[i]] = 0
    for i in range(len(event_b_starts)):
        b[event_b_starts[i]:even_b_ends[i]] = 1
    # Intersection
    commonEvents = np.logical_and(a, b)
    # Process
    numEvents, eventStarts, eventEnds, eventIII, eventDur, eventIBI = compute_basic_features(commonEvents, 0.5, 0, np.inf, framesBeforeDetection, framesAfterDetection, Fs)
    return commonEvents, numEvents, eventStarts, eventEnds, eventIII, eventDur, eventIBI