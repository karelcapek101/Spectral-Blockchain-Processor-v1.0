from smrk_processor import SpectralBlockchainProcessor

# Inicializace procesoru
processor = SpectralBlockchainProcessor(N=32)  
# Simulace přidání bloků
new_block1 = np.random.normal(0, 0.01, 32)  # Simulace transakcí
processor.add_block(new_block1)

new_block2 = np.random.normal(0, 0.01, 32)
processor.add_block(new_block2)

# Simulace forků a interference
fork1 = processor.state.copy() + np.random.normal(0, 0.05, 32)
fork2 = processor.state.copy() + np.random.normal(0, 0.05, 32)
processor.fork_interference([fork1, fork2])

# Výpočet invariantu
trace = processor.compute_trace_invariant()
print(f"Finální trace invariant: {trace}")

# Audit log
print("\nAudit Log (pro replay):")
print(processor.get_audit_log())
