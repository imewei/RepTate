import numpy as np

def timeArraySplit(t, Tend):
    """Split a time array into two overlapping segments at a specified time.

    Divides a time array into two parts: one from the start to Tend, and one from
    Tend to the end. Both arrays include Tend as a boundary point, creating an
    overlap at the split point.

    Args:
        t (np.ndarray): Input time array to be split (must be sorted in ascending order).
        Tend (float): Time value at which to split the array.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple (t1, t2) where t1 contains times from
            start to Tend (inclusive) and t2 contains times from Tend (inclusive) to end.
    """
    lastPoint= np.nonzero(t >= Tend)[0][0]

    t1=t[:lastPoint]
    #t1=np.append([0],t[:lastPoint])
    t1=np.append(t1,[Tend])

    t2=np.append([Tend],t[lastPoint:])

    return t1,t2
    
