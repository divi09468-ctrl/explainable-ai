from response_module import (
    calculate_risk,
    simulate_block,
    get_simulated_blocklist
)

# Example risk calculation
result = calculate_risk(
    attack_probability=0.90,
    attack_type="dos",
    asset_criticality=0.80
)

print("Risk assessment:")
print(result)

# Demonstrate application-only blocking
block_result = simulate_block("192.0.2.10")

print("\nSimulated response:")
print(block_result)

print("\nApplication blocklist:")
print(get_simulated_blocklist())