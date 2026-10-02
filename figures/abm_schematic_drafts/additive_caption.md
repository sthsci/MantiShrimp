# Additive mechanics, 2026-10-01

**Mechanical interactions and synapse dynamics.** Each living cell pair has a
Hookean overlap-repulsion contribution of magnitude K_rep max(L−δ, 0), where
δ is centre distance and L is the sum of radii. A bound killer–target pair
also experiences attraction of magnitude F_A whenever δ<R_cap, including
during overlap. The signed net force on a bound overlapping pair is therefore
K_rep(L−δ)−F_A; positive and negative values denote repulsion and attraction,
respectively. Non-overlapping unbound pairs have zero pair force. Binding can
occur only within δ≤L+ε, with per-step probability 1−exp(−k_bind Δt).
Existing synapses dissociate with probability 1−exp(−k_unbind Δt), or end when
δ>R_cap or the target dies. Here ε=0.05L and R_cap=1.5L. Force amplitudes are
schematic; the curves are evaluated using the updated simulator.

The earlier `panel_b_*.png`, `abm_figure_integrated.*`, `mechanics_insert.png`
and `force_and_binding_rules.*` images show the previous force rule and should
not be used to illustrate the updated simulator. The new outputs are
`force_and_binding_rules_additive.png` and `.svg`.
