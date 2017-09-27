"""Little helper functions
"""
import numpy as np

def is_string_like(obj):
    """Return true if obj behaves like a string"""
    try:
        obj + ''
    except:
        return False
    return True

def is_number_like(obj):
    """Return True if obj behaves like a SINGLE number."""
    try:
        obj = obj + 1 # might still be an array!
        obj = float(obj)
    except:
        return False
    return True

def get_nfft(nfft, g):
    """Helper to get a good value for nfft for xcorr_grouped_df
    and similar functions.
    
    Parameters:
    -----------
    
        nfft: str, int
            'pad':   Smallest number of two larger than the smallest group.
            'demix': A factor two more than pad, allowing to perfectly demix
                     causal and anticausal parts.
            'len':   Exactly the smllest group size.
            'crop':  Largest number of two smaller then the smallest group size.
            int:     Pass through
            
            If nfft is string-like and contains '>', it will be split at that
            point and what follows is converted to int and use as a minimum
            for the smallest group size considered. I.e. 'len > 100' will set
            nfft to the length of the smallest group that contains more than
            100 events.
        g: pandas GroupBy object
            The grouped DataFrame to analyse
                     
    Returns:
    --------
        
        nfft, events_required. The latter corresponds to a 
        cutoff size below which groups should be discarded. The largest
        valid lag without is min(nfft/2, events_required).
    """
    events_required = 4 # arbitrary initialisation, never returned
    if is_string_like(nfft):
        if '>' in nfft:
            # make sure we can fill
            events_required = int(nfft.split('>')[1])
            assert int(g.count().max().max()) > events_required, (
                "Not enough events per day!"
            )
        n = np.max([events_required, int(g.count().min().min())])
        events_required = n
        if 'pad' in nfft or 'auto' in nfft:
            # smallest power of two > n
            nfft = int(2**np.ceil(np.log2(n)))
            #events_required = nfft / 2
        elif 'demix' in nfft:
            # smallest power of two for perfect demix
            nfft = int(2**np.ceil(np.log2(n)+1))
        elif 'len' in nfft:
            nfft = n
        elif 'crop' in nfft:
            # largest power of two < n
            nfft = int(2**np.floor(np.log2(n)))
        else:
            raise ValueError, "Can't understand nfft='%s'" % nfft
    else:
        nfft = int(nfft)
        events_required = int(np.ceil(nfft / 2.))
    return nfft, events_required
