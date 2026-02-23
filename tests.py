import numpy as np
from smrk_processor import SpectralBlockchainProcessor

def test_processor():
    processor = SpectralBlockchainProcessor(N=16)
    assert len(processor.state) == 16, "State size mismatch"
    
    H_state = processor.apply_hamiltonian()
    assert np.allclose(np.mean(H_state), np.mean(processor.state), atol=1e-1), "Hamiltonian application failed"
    
    trace = processor.compute_trace_invariant()
    assert abs(trace) > 0, "Trace invariant zero"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_processor()
